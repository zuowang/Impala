// Copyright 2012 Cloudera Inc.
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
import java.util.List;

import com.cloudera.impala.catalog.Db;
import com.cloudera.impala.catalog.Function.CompareMode;
import com.cloudera.impala.catalog.ScalarFunction;
import com.cloudera.impala.catalog.Type;
import com.cloudera.impala.common.AnalysisException;
import com.cloudera.impala.thrift.TExprNode;
import com.cloudera.impala.thrift.TExprNodeType;
import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

/**
 * &&, ||, ! predicates.
 *
 */
public class CompoundPredicate extends Predicate {
  public enum Operator {
    AND("AND"),
    OR("OR"),
    NOT("NOT");

    private final String description;

    private Operator(String description) {
      this.description = description;
    }

    @Override
    public String toString() {
      return description;
    }
  }
  private final Operator op_;

  public static void initBuiltins(Db db) {
    // AND and OR are implemented as custom exprs, so they do not have a function symbol.
    db.addBuiltin(ScalarFunction.createBuiltinOperator(
        Operator.AND.name(), "",
        Lists.<Type>newArrayList(Type.BOOLEAN, Type.BOOLEAN), Type.BOOLEAN));
    db.addBuiltin(ScalarFunction.createBuiltinOperator(
        Operator.OR.name(), "",
        Lists.<Type>newArrayList(Type.BOOLEAN, Type.BOOLEAN), Type.BOOLEAN));
    db.addBuiltin(ScalarFunction.createBuiltinOperator(
        Operator.NOT.name(), "impala::CompoundPredicate::Not",
        Lists.<Type>newArrayList(Type.BOOLEAN), Type.BOOLEAN));
  }

  public CompoundPredicate(Operator op, Expr e1, Expr e2) {
    super();
    this.op_ = op;
    Preconditions.checkNotNull(e1);
    children_.add(e1);
    Preconditions.checkArgument(op == Operator.NOT && e2 == null
        || op != Operator.NOT && e2 != null);
    if (e2 != null) children_.add(e2);
  }

  /**
   * Copy c'tor used in clone().
   */
  protected CompoundPredicate(CompoundPredicate other) {
    super(other);
    op_ = other.op_;
  }

  public Operator getOp() { return op_; }

  @Override
  public boolean equals(Object obj) {
    if (!super.equals(obj)) return false;
    return ((CompoundPredicate) obj).op_ == op_;
  }

  @Override
  public String debugString() {
    return Objects.toStringHelper(this)
        .add("op", op_)
        .addValue(super.debugString())
        .toString();
  }

  @Override
  public String toSqlImpl() {
    if (children_.size() == 1) {
      Preconditions.checkState(op_ == Operator.NOT);
      return "NOT " + getChild(0).toSql();
    } else {
      return getChild(0).toSql() + " " + op_.toString() + " " + getChild(1).toSql();
    }
  }

  @Override
  protected void toThrift(TExprNode msg) {
    msg.node_type = TExprNodeType.COMPOUND_PRED;
  }

  @Override
  public void analyze(Analyzer analyzer) throws AnalysisException {
    if (isAnalyzed_) return;
    super.analyze(analyzer);

    // Check that children are predicates.
    for (Expr e: children_) {
      if (!e.getType().isBoolean() && !e.getType().isNull()) {
        throw new AnalysisException(String.format("Operand '%s' part of predicate " +
            "'%s' should return type 'BOOLEAN' but returns type '%s'.",
            e.toSql(), toSql(), e.getType().toSql()));
      }
    }

    fn_ = getBuiltinFunction(analyzer, op_.toString(), collectChildReturnTypes(),
        CompareMode.IS_NONSTRICT_SUPERTYPE_OF);
    Preconditions.checkState(fn_ != null);
    Preconditions.checkState(fn_.getReturnType().isBoolean());
    castForFunctionCall(false);

    if (!getChild(0).hasSelectivity() ||
        (children_.size() == 2 && !getChild(1).hasSelectivity())) {
      // Give up if one of our children has an unknown selectivity.
      selectivity_ = -1;
      return;
    }

    switch (op_) {
      case AND:
        selectivity_ = getChild(0).selectivity_ * getChild(1).selectivity_;
        break;
      case OR:
        selectivity_ = getChild(0).selectivity_ + getChild(1).selectivity_
            - getChild(0).selectivity_ * getChild(1).selectivity_;
        break;
      case NOT:
        selectivity_ = 1.0 - getChild(0).selectivity_;
        break;
    }
    selectivity_ = Math.max(0.0, Math.min(1.0, selectivity_));
  }

  /**
   * Retrieve the slots bound by BinaryPredicate, InPredicate and
   * CompoundPredicates in the subtree rooted at 'this'.
   */
  public ArrayList<SlotRef> getBoundSlots() {
    ArrayList<SlotRef> slots = Lists.newArrayList();
    for (int i = 0; i < getChildren().size(); ++i) {
      if (getChild(i) instanceof BinaryPredicate ||
          getChild(i) instanceof InPredicate) {
        slots.add(((Predicate)getChild(i)).getBoundSlot());
      } else if (getChild(i) instanceof CompoundPredicate) {
        slots.addAll(((CompoundPredicate)getChild(i)).getBoundSlots());
      }
    }
    return slots;
  }

  /**
   * Negates a CompoundPredicate.
   */
  @Override
  public Expr negate() {
    if (op_ == Operator.NOT) return getChild(0);
    Expr negatedLeft = getChild(0).negate();
    Expr negatedRight = getChild(1).negate();
    Operator newOp = (op_ == Operator.OR) ? Operator.AND : Operator.OR;
    return new CompoundPredicate(newOp, negatedLeft, negatedRight);
  }

  /**
   * Creates a conjunctive predicate from a list of exprs.
   */
  public static Expr createConjunctivePredicate(List<Expr> conjuncts) {
    Expr conjunctivePred = null;
    for (Expr expr: conjuncts) {
      if (conjunctivePred == null) {
        conjunctivePred = expr;
        continue;
      }
      conjunctivePred = new CompoundPredicate(CompoundPredicate.Operator.AND,
          expr, conjunctivePred);
    }
    return conjunctivePred;
  }

  @Override
  public Expr clone() { return new CompoundPredicate(this); }

  // Create an AND predicate between two exprs, 'lhs' and 'rhs'. If
  // 'rhs' is null, simply return 'lhs'.
  public static Expr createConjunction(Expr lhs, Expr rhs) {
    if (rhs == null) return lhs;
    return new CompoundPredicate(Operator.AND, rhs, lhs);
  }
}
