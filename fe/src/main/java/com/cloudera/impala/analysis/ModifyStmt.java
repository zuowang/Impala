// Copyright 2015 Cloudera Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.cloudera.impala.analysis;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import com.cloudera.impala.authorization.Privilege;
import com.cloudera.impala.authorization.PrivilegeRequestBuilder;
import com.cloudera.impala.catalog.Column;
import com.cloudera.impala.catalog.KuduTable;
import com.cloudera.impala.catalog.Table;
import com.cloudera.impala.catalog.Type;
import com.cloudera.impala.common.AnalysisException;
import com.cloudera.impala.common.Pair;
import com.cloudera.impala.planner.DataSink;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import org.slf4j.LoggerFactory;

import static java.lang.String.format;

/**
 * Abstract super class for statements that modify existing data like
 * UPDATE and DELETE.
 *
 * The ModifyStmt has four major parts:
 *   - targetTablePath (not null)
 *   - fromClause (not null)
 *   - assignmentExprs (not null, can be empty)
 *   - wherePredicate (nullable)
 *
 * In the analysis phase, a SelectStmt is created with the result expressions set to
 * match the right-hand side of the assignments in addition to projecting the key columns
 * of the underlying table. During query execution, the plan that
 * is generated from this SelectStmt produces all rows that need to be modified.
 *
 * Currently, only Kudu tables can be modified.
 */
public abstract class ModifyStmt extends StatementBase {
  private final static org.slf4j.Logger LOG = LoggerFactory.getLogger(ModifyStmt.class);

  // List of explicitly mentioned assignment expressions in the UPDATE's SET clause
  protected final List<Pair<SlotRef, Expr>> assignments_;

  // Optional WHERE clause of the statement
  protected final Expr wherePredicate_;

  // Path identifying the target table.
  protected final List<String> targetTablePath_;

  // TableRef identifying the target table, set during analysis.
  protected TableRef targetTableRef_;

  protected FromClause fromClause_;

  // Result of the analysis of the internal SelectStmt that produces the rows that
  // will be modified.
  protected SelectStmt sourceStmt_;

  // Target Kudu table. Since currently only Kudu tables are supported, we use a
  // concrete table class. Result of analysis.
  protected KuduTable table_;

  // Position mapping of output expressions of the sourceStmt_ to column indices in the
  // target table. The i'th position in this list maps to the referencedColumns_[i]'th
  // position in the target table. Set in createSourceStmt() during analysis.
  protected ArrayList<Integer> referencedColumns_;

  // END: Members that need to be reset()
  /////////////////////////////////////////

  // On tables with a primary key, ignore key not found errors.
  protected final boolean ignoreNotFound_;

  public ModifyStmt(List<String> targetTablePath, FromClause fromClause,
      List<Pair<SlotRef, Expr>> assignmentExprs,
      Expr wherePredicate, boolean ignoreNotFound) {
    targetTablePath_ = Preconditions.checkNotNull(targetTablePath);
    fromClause_ = Preconditions.checkNotNull(fromClause);
    assignments_ = Preconditions.checkNotNull(assignmentExprs);
    wherePredicate_ = wherePredicate;
    ignoreNotFound_ = ignoreNotFound;
  }

  /**
   * The analysis of the ModifyStmt proceeds as follows: First, the FROM clause is
   * analyzed and the targetTablePath is verified to be a valid alias into the FROM
   * clause. When the target table is identified, the assignment expressions are
   * validated and as a last step the internal SelectStmt is produced and analyzed.
   * Potential query rewrites for the select statement are implemented here and are not
   * triggered externally by the statement rewriter.
   */
  @Override
  public void analyze(Analyzer analyzer) throws AnalysisException {
    super.analyze(analyzer);
    fromClause_.analyze(analyzer);

    List<Path> candidates = analyzer.getTupleDescPaths(targetTablePath_);
    if (candidates.isEmpty()) {
      throw new AnalysisException(format("'%s' is not a valid table alias or reference.",
          Joiner.on(".").join(targetTablePath_)));
    }

    Preconditions.checkState(candidates.size() == 1);
    Path path = candidates.get(0);
    path.resolve();

    if (path.destTupleDesc() == null) {
      throw new AnalysisException(format(
          "'%s' is not a table alias. Using the FROM clause requires the target table " +
              "to be a table alias.",
          Joiner.on(".").join(targetTablePath_)));
    }

    targetTableRef_ = analyzer.getTableRef(path.getRootDesc().getId());
    if (targetTableRef_ instanceof InlineViewRef) {
      throw new AnalysisException(format("Cannot modify view: '%s'",
          targetTableRef_.toSql()));
    }

    Preconditions.checkNotNull(targetTableRef_);
    Table dstTbl = targetTableRef_.getTable();
    // Only Kudu tables can be updated
    if (!(dstTbl instanceof KuduTable)) {
      throw new AnalysisException(
          format("Impala does not support modifying a non-Kudu table: %s",
              dstTbl.getFullName()));
    }
    table_ = (KuduTable) dstTbl;

    // Make sure that the user is allowed to modify the target table, since no
    // UPDATE / DELETE privilege exists, we reuse the INSERT one.
    analyzer.registerPrivReq(new PrivilegeRequestBuilder()
        .onTable(table_.getDb().getName(), table_.getName())
        .allOf(Privilege.INSERT).toRequest());

    // Validates the assignments_ and creates the sourceStmt_.
    if (sourceStmt_ == null) createSourceStmt(analyzer);
    sourceStmt_.analyze(analyzer);
  }

  @Override
  public void reset() {
    super.reset();
    fromClause_.reset();
    if (sourceStmt_ != null) sourceStmt_.reset();
    table_ = null;
    referencedColumns_.clear();
  }

  /**
   * Builds and validates the sourceStmt_. The select list of the sourceStmt_ contains
   * first the SlotRefs for the key Columns, followed by the expressions representing the
   * assignments. This method sets the member variables for the sourceStmt_ and the
   * referencedColumns_.
   *
   * This is only run once, on the first analysis. Following analysis will reset() and
   * reuse previously created statements.
   */
  private void createSourceStmt(Analyzer analyzer)
      throws AnalysisException {
    // Builds the select list and column position mapping for the target table.
    ArrayList<SelectListItem> selectList = Lists.newArrayList();
    referencedColumns_ = Lists.newArrayList();
    buildAndValidateAssignmentExprs(analyzer, selectList, referencedColumns_);

    // Analyze the generated select statement.
    sourceStmt_ = new SelectStmt(new SelectList(selectList), fromClause_, wherePredicate_,
        null, null, null, null);

    // cast result expressions to the correct type of the referenced slot of the
    // target table
    int keyColumnsOffset = table_.getKuduKeyColumnNames().size();
    for (int i = keyColumnsOffset; i < sourceStmt_.resultExprs_.size(); ++i) {
      sourceStmt_.resultExprs_.set(i, sourceStmt_.resultExprs_.get(i).castTo(
          assignments_.get(i - keyColumnsOffset).first.getType()));
    }
  }

  /**
   * Validates the list of value assignments that should be used to modify the target
   * table. It verifies that only those columns are referenced that belong to the target
   * table, no key columns are modified, and that a single column is not modified multiple
   * times. Analyzes the Exprs and SlotRefs of assignments_ and writes a list of
   * SelectListItems to the out parameter selectList that is used to build the select list
   * for sourceStmt_. A list of integers indicating the column position of an entry in the
   * select list in the target table is written to the out parameter referencedColumns.
   *
   * In addition to the expressions that are generated for each assignment, the
   * expression list contains an expression for each key column. The key columns
   * are always prepended to the list of expression representing the assignments.
   */
  private void buildAndValidateAssignmentExprs(Analyzer analyzer,
      ArrayList<SelectListItem> selectList, ArrayList<Integer> referencedColumns)
      throws AnalysisException {
    // The order of the referenced columns equals the order of the result expressions
    HashSet<SlotId> uniqueSlots = Sets.newHashSet();
    HashSet<SlotId> keySlots = Sets.newHashSet();

    // Mapping from column name to index
    ArrayList<Column> cols = table_.getColumnsInHiveOrder();
    HashMap<String, Integer> colIndexMap = Maps.newHashMap();
    for (int i = 0; i < cols.size(); i++) {
      colIndexMap.put(cols.get(i).getName(), i);
    }

    // Add the key columns as slot refs
    for (String k : table_.getKuduKeyColumnNames()) {
      ArrayList<String> path = Path.createRawPath(targetTableRef_.getUniqueAlias(), k);
      SlotRef ref = new SlotRef(path);
      ref.analyze(analyzer);
      selectList.add(new SelectListItem(ref, null));
      uniqueSlots.add(ref.getSlotId());
      keySlots.add(ref.getSlotId());
      referencedColumns.add(colIndexMap.get(k));
    }

    // Assignments are only used in the context of updates.
    for (Pair<SlotRef, Expr> valueAssignment : assignments_) {
      Expr rhsExpr = valueAssignment.second;
      rhsExpr.analyze(analyzer);

      SlotRef lhsSlotRef = valueAssignment.first;
      lhsSlotRef.analyze(analyzer);

      // Correct target table
      if (!lhsSlotRef.isBoundByTupleIds(targetTableRef_.getId().asList())) {
        throw new AnalysisException(
            format("Left-hand side column '%s' in assignment expression '%s=%s' does not "
                + "belong to target table '%s'", lhsSlotRef.toSql(), lhsSlotRef.toSql(),
                rhsExpr.toSql(), targetTableRef_.getDesc().getTable().getFullName()));
      }

      // No subqueries for rhs expression
      if (rhsExpr.contains(Subquery.class)) {
        throw new AnalysisException(
            format("Subqueries are not supported as update expressions for column '%s'",
                lhsSlotRef.toSql()));
      }

      Column c = lhsSlotRef.getResolvedPath().destColumn();
      // TODO(Kudu) Add test for this code-path when Kudu supports nested types
      if (c == null) {
        throw new AnalysisException(
            format("Left-hand side in assignment expression '%s=%s' must be a column " +
                "reference", lhsSlotRef.toSql(), rhsExpr.toSql()));
      }

      if (keySlots.contains(lhsSlotRef.getSlotId())) {
        throw new AnalysisException(format("Key column '%s' cannot be updated.",
            lhsSlotRef.toSql()));
      }

      if (uniqueSlots.contains(lhsSlotRef.getSlotId())) {
        throw new AnalysisException(
            format("Duplicate value assignment to column: '%s'", lhsSlotRef.toSql()));
      }

      rhsExpr = checkTypeCompatibility(c, rhsExpr);
      uniqueSlots.add(lhsSlotRef.getSlotId());
      selectList.add(new SelectListItem(rhsExpr, null));
      referencedColumns.add(colIndexMap.get(c.getName()));
    }
  }

  /**
   * Checks for type compatibility of column and expr.
   * Returns compatible (possibly cast) expr.
   * TODO(kudu-merge) Find a way to consolidate this with
   * InsertStmt#checkTypeCompatibility()
   */
  private Expr checkTypeCompatibility(Column column, Expr expr)
      throws AnalysisException {
    // Check for compatible type, and add casts to the selectListExprs if necessary.
    // We don't allow casting to a lower precision type.
    Type colType = column.getType();
    Type exprType = expr.getType();
    // Trivially compatible, unless the type is complex.
    if (colType.equals(exprType) && !colType.isComplexType()) return expr;

    Type compatibleType = Type.getAssignmentCompatibleType(colType, exprType, false);
    // Incompatible types.
    if (!compatibleType.isValid()) {
      throw new AnalysisException(
          format("Target table '%s' is incompatible with source expressions.\nExpression "
              + "'%s' (type: %s) is not compatible with column '%s' (type: %s)",
              targetTableRef_.getDesc().getTable().getFullName(), expr.toSql(),
              exprType.toSql(), column.getName(), colType.toSql()));
    }
    // Loss of precision when inserting into the table.
    if (!compatibleType.equals(colType) && !compatibleType.isNull()) {
      throw new AnalysisException(
          format("Possible loss of precision for target table '%s'.\n" +
                  "Expression '%s' (type: %s) would need to be cast to %s" +
                  " for column '%s'",
              targetTableRef_.getDesc().getTable().getFullName(), expr.toSql(),
              exprType.toSql(), colType.toSql(), column.getName()));
    }
    // Add a cast to the selectListExpr to the higher type.
    return expr.castTo(compatibleType);
  }


  public QueryStmt getQueryStmt() { return sourceStmt_; }
  public abstract DataSink createDataSink();
  public abstract String toSql();


}
