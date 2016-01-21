// Copyright 2013 Cloudera Inc.
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

#ifndef IMPALA_UTIL_SPINLOCK_H
#define IMPALA_UTIL_SPINLOCK_H

#include <cpuid.h>

#include "common/atomic.h"
#include "common/logging.h"


namespace impala {

#define __rtm_force_inline __attribute__((__always_inline__)) inline
#define _XBEGIN_STARTED		(~0u)
#define _XABORT_EXPLICIT	(1 << 0)
#define _XABORT_RETRY		(1 << 1)
#define _XABORT_CONFLICT	(1 << 2)
#define _XABORT_CAPACITY	(1 << 3)
#define _XABORT_DEBUG		(1 << 4)
#define _XABORT_NESTED		(1 << 5)
#define _XABORT_CODE(x)		(((x) >> 24) & 0xff)

static __rtm_force_inline int _xbegin(void) {
	int ret = _XBEGIN_STARTED;
	asm volatile(".byte 0xc7,0xf8 ; .long 0" : "+a" (ret) :: "memory");
	return ret;
}

static __rtm_force_inline void _xend(void) {
	 asm volatile(".byte 0x0f,0x01,0xd5" ::: "memory");
}

// This is a macro because some compilers do not propagate the constant
// through an inline with optimization disabled.
#define _xabort(status) \
	asm volatile(".byte 0xc6,0xf8,%P0" :: "i" (status) : "memory")

static __rtm_force_inline int _xtest(void) {
	unsigned char out;
	asm volatile(".byte 0x0f,0x01,0xd6 ; setnz %0" : "=r" (out) :: "memory");
	return out;
}

/// Lightweight spinlock.
class SpinLock {
 public:
  SpinLock() : locked_(false) {}

#ifdef IR_COMPILE
  void lock() {
    if (!try_lock()) SlowAcquire();
  }
#else
  static const int CPUID_RTM = 1 << 11;
  static const int CPUID_HLE = 1 << 4;

  static inline int cpu_has_rtm() {
    if (__get_cpuid_max(0, NULL) >= 7) {
      unsigned a, b, c, d;
      __cpuid_count(7, 0, a, b, c, d);
      return !!(b & CPUID_RTM);
    }
    return 0;
  }

  static inline int cpu_has_hle() __attribute__((unused)) {
    if (__get_cpuid_max(0, NULL) >= 7) {
      unsigned a, b, c, d;
      __cpuid_count(7, 0, a, b, c, d);
      return !!(b & CPUID_HLE);
    }
    return 0;
  }

  typedef void (impala::SpinLock::*FP)();

  FP lock_ifunc() __attribute__((used)) {
    if (cpu_has_rtm()) {
      return &SpinLock::rtm_lock;
    } else if (cpu_has_hle()) {
      return &SpinLock::hle_lock;
    } else {
      return &SpinLock::normal_lock;
    }
  }

  void lock() __attribute__((ifunc ("_ZN6impala8SpinLock10lock_ifuncEv")));

  /// Acquires the lock, spins until the lock becomes available
  void normal_lock() {
    if (!try_lock()) SlowAcquire();
  }

  void rtm_lock() {
    unsigned int tm_status = 0;
  tm_try:
    if ((tm_status = _xbegin()) == _XBEGIN_STARTED) {
      if (!locked_) return;
      // Otherwise fall back to the spinlock by aborting.
      _xabort(0xff); // 0xff canonically denotes 'lock is taken'.
    } else {
      // _xbegin could have had a conflict, been aborted, etc.
      if (tm_status & _XABORT_RETRY) {
        AtomicUtil::CpuWait();
        goto tm_try; // Retry
      }
      if (tm_status & _XABORT_EXPLICIT) {
        int code = _XABORT_CODE(tm_status);
        if (code == 0xff) goto tm_fail; // Lock was taken; fallback
      }
  tm_fail:
      normal_lock();
    }
  }

  void hle_lock() {
    while (__sync_lock_test_and_set(&locked_, true)) {
      do {
        if (!__sync_val_compare_and_swap(&locked_, true, true)) break;
        for (int i = 0; i < NUM_SPIN_CYCLES; ++i) {
          AtomicUtil::CpuWait();
        }
        if (!__sync_val_compare_and_swap(&locked_, true, true)) break;
        sched_yield();
      } while (true);
    }
  }
#endif

#ifdef IR_COMPILE
  void unlock() {
    // Memory barrier here. All updates before the unlock need to be made visible.
    __sync_synchronize();
    DCHECK(locked_);
    locked_ = false;
  }
#else
  FP unlock_ifunc() __attribute__((used)) {
    if (cpu_has_rtm()) {
      return &SpinLock::rtm_unlock;
    } else if (cpu_has_hle()) {
      return &SpinLock::hle_unlock;
    } else {
      return &SpinLock::normal_unlock;
    }
  }

  void unlock() __attribute__((ifunc ("_ZN6impala8SpinLock12unlock_ifuncEv")));

  void normal_unlock() {
    // Memory barrier here. All updates before the unlock need to be made visible.
    __sync_synchronize();
    DCHECK(locked_);
    locked_ = false;
  }

  void rtm_unlock() {
    if (!locked_) {
      _xend(); // Commit transaction
    } else {
      // Otherwise, the lock was taken by us, so release it too.
      normal_unlock();
    }
  }

  void hle_unlock() {
    __sync_lock_release(&locked_);
  }
#endif

  /// Tries to acquire the lock
  inline bool try_lock() { return __sync_bool_compare_and_swap(&locked_, false, true); }

  void DCheckLocked() { DCHECK(locked_); }

 private:

  /// Out-of-line definition of the actual spin loop. The primary goal is to have the
  /// actual lock method as short as possible to avoid polluting the i-cache with
  /// unnecessary instructions in the non-contested case.
  void SlowAcquire();

  /// In typical spin lock implements, we want to spin (and keep the core fully busy),
  /// for some number of cycles before yielding. Consider these three cases:
  ///  1) lock is un-contended - spinning doesn't kick in and has no effect.
  ///  2) lock is taken by another thread and that thread finishes quickly.
  ///  3) lock is taken by another thread and that thread is slow (e.g. scheduled away).
  ///
  /// In case 3), we'd want to yield so another thread can do work. This thread
  /// won't be able to do anything useful until the thread with the lock runs again.
  /// In case 2), we don't want to yield (and give up our scheduling time slice)
  /// since we will get to run soon after.
  /// To try to get the best of everything, we will busy spin for a while before
  /// yielding to another thread.
  /// TODO: how do we set this.
  static const int NUM_SPIN_CYCLES = 70;
  /// TODO: pad this to be a cache line?
  bool locked_;
};

}
#endif
