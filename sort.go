// Package sort provides primitives for sorting slices using golang generics
package sort

import (
	"sort"
)

// LessFunc is the signature for the function to determine which element is the less.
type LessFunc func(int, int) bool

var _ sort.Interface = (*SortableSlice[any])(nil)

// SortableSlice is an implementation of sort.Interface for a given slice.
type SortableSlice[T any] struct {
	// Data is the slice of any types.
	Data []T

	// LessFunc reports whether the element with index i
	// must sort before the element with index j.
	//
	// If both Less(i, j) and Less(j, i) are false,
	// then the elements at index i and j are considered equal.
	// Sort may place equal elements in any order in the final result,
	// while Stable preserves the original input order of equal elements.
	//
	// Less must describe a transitive ordering:
	//  - if both Less(i, j) and Less(j, k) are true, then Less(i, k) must be true as well.
	//  - if both Less(i, j) and Less(j, k) are false, then Less(i, k) must be false as well.
	//
	// Note that floating-point comparison (the < operator on float32 or float64 values)
	// is not a transitive ordering when not-a-number (NaN) values are involved.
	// See Float64Slice.Less for a correct implementation for floating-point values.
	LessFunc LessFunc
}

// New will create a new SortableSlice.
func New[T any](x []T, less LessFunc) *SortableSlice[T] {
	return &SortableSlice[T]{
		Data:     x,
		LessFunc: less,
	}
}

// Len is the number of elements in the collection.
func (s SortableSlice[T]) Len() int {
	return len(s.Data)
}

// Less will call LessFunc of SortableSlice.
func (s SortableSlice[T]) Less(i, j int) bool {
	return s.LessFunc(i, j)
}

// Swap swaps the elements with indexes i and j.
func (s SortableSlice[T]) Swap(i, j int) {
	s.Data[i], s.Data[j] = s.Data[j], s.Data[i]
}

// Sort sorts data in ascending order as determined by the Less method.
// It makes one call to data.Len to determine n and O(n*log(n)) calls to
// data.Less and data.Swap. The sort is not guaranteed to be stable.
func Sort[T any](data []T, less LessFunc) {
	sort.Sort(New(data, less))
}

// IsSorted reports whether data is sorted.
func IsSorted[T any](data []T, less LessFunc) {
	sort.IsSorted(New(data, less))
}

// Reverse returns the reverse order for data.
func Reverse[T any](data []T, less LessFunc) []T {
	if ss, ok := sort.Reverse(New(data, less)).(SortableSlice[T]); ok {
		return ss.Data
	}

	return nil
}

// Notes on stable sorting:
// The used algorithms are simple and provable correct on all input and use
// only logarithmic additional stack space. They perform well if compared
// experimentally to other stable in-place sorting algorithms.
//
// Remarks on other algorithms evaluated:
//  - GCC's 4.6.3 stable_sort with merge_without_buffer from libstdc++:
//    Not faster.
//  - GCC's __rotate for block rotations: Not faster.
//  - "Practical in-place mergesort" from  Jyrki Katajainen, Tomi A. Pasanen
//    and Jukka Teuhola; Nordic Journal of Computing 3,1 (1996), 27-40:
//    The given algorithms are in-place, number of Swap and Assignments
//    grow as n log n but the algorithm is not stable.
//  - "Fast Stable In-Place Sorting with O(n) Data Moves" J.I. Munro and
//    V. Raman in Algorithmica (1996) 16, 115-160:
//    This algorithm either needs additional 2n bits or works only if there
//    are enough different elements available to encode some permutations
//    which have to be undone later (so not stable on any input).
//  - All the optimal in-place sorting/merging algorithms I found are either
//    unstable or rely on enough different elements in each step to encode the
//    performed block rearrangements. See also "In-Place Merging Algorithms",
//    Denham Coates-Evely, Department of Computer Science, Kings College,
//    January 2004 and the references in there.
//  - Often "optimal" algorithms are optimal in the number of assignments
//    but Interface has only Swap as operation.

/*
Complexity of Stable Sorting


Complexity of block swapping rotation

Each Swap puts one new element into its correct, final position.
Elements which reach their final position are no longer moved.
Thus block swapping rotation needs |u|+|v| calls to Swaps.
This is best possible as each element might need a move.

Pay attention when comparing to other optimal algorithms which
typically count the number of assignments instead of swaps:
E.g. the optimal algorithm of Dudzinski and Dydek for in-place
rotations uses O(u + v + gcd(u,v)) assignments which is
better than our O(3 * (u+v)) as gcd(u,v) <= u.


Stable sorting by SymMerge and BlockSwap rotations

SymMerg complexity for same size input M = N:
Calls to Less:  O(M*log(N/M+1)) = O(N*log(2)) = O(N)
Calls to Swap:  O((M+N)*log(M)) = O(2*N*log(N)) = O(N*log(N))

(The following argument does not fuzz over a missing -1 or
other stuff which does not impact the final result).

Let n = data.Len(). Assume n = 2^k.

Plain merge sort performs log(n) = k iterations.
On iteration i the algorithm merges 2^(k-i) blocks, each of size 2^i.

Thus iteration i of merge sort performs:
Calls to Less  O(2^(k-i) * 2^i) = O(2^k) = O(2^log(n)) = O(n)
Calls to Swap  O(2^(k-i) * 2^i * log(2^i)) = O(2^k * i) = O(n*i)

In total k = log(n) iterations are performed; so in total:
Calls to Less O(log(n) * n)
Calls to Swap O(n + 2*n + 3*n + ... + (k-1)*n + k*n)
   = O((k/2) * k * n) = O(n * k^2) = O(n * log^2(n))


Above results should generalize to arbitrary n = 2^k + p
and should not be influenced by the initial insertion sort phase:
Insertion sort is O(n^2) on Swap and Less, thus O(bs^2) per block of
size bs at n/bs blocks:  O(bs*n) Swaps and Less during insertion sort.
Merge sort iterations start at i = log(bs). With t = log(bs) constant:
Calls to Less O((log(n)-t) * n + bs*n) = O(log(n)*n + (bs-t)*n)
   = O(n * log(n))
Calls to Swap O(n * log^2(n) - (t^2+t)/2*n) = O(n * log^2(n))

*/

// Stable sorts data in ascending order as determined by the Less method,
// while keeping the original order of equal elements.
//
// It makes one call to data.Len to determine n, O(n*log(n)) calls to
// data.Less and O(n*log(n)*log(n)) calls to data.Swap.
func Stable[T any](data []T, less LessFunc) {
	sort.Stable(New(data, less))
}

// Slice sorts the slice x using the provided less function.
//
// The sort is not guaranteed to be stable: equal elements
// may be reversed from their original order.
// For a stable sort, use SliceStable.
//
// The less function must satisfy the same requirements as
// the SortableSlice type's LessFunc.
func Slice[T any](x []T, less LessFunc) {
	Sort(x, less)
}

// SliceStable sorts the slice x using the provided less
// function, keepint equal elements in their original order.
//
// The less function must satisfy the same requirements as
// the SortableSlice type's LessFunc.
func SliceStable[T any](x []T, less LessFunc) {
	Stable(x, less)
}

// IntSlice sorts a slice of ints in increasing order.
func IntSlice(x []int) {
	Slice(x, func(i, j int) bool {
		return x[i] < x[j]
	})
}

// IntSlice stable sorts a slice of ints in increasing order.
func IntSliceStable(x []int) {
	Stable(x, func(i, j int) bool {
		return x[i] < x[j]
	})
}

// StringSlice sorts a slice of strings in increasing order.
func StringSlice(x []string) {
	Slice(x, func(i, j int) bool {
		return x[i] < x[j]
	})
}

// StringSliceStable stable sorts a slice of strings in increasing
// order.
func StringSliceStable(x []string) {
	Stable(x, func(i, j int) bool {
		return x[i] < x[j]
	})
}
