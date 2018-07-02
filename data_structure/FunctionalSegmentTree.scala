package com.FunctionalDataStructure

trait Monoid[T] {
  def zero: T
  def add(a: T, b: T): T
}

object Monoid {
  // Make sure binary operation is associative
  implicit val IntMonoid = new Monoid[Int] {
    def add(a: Int, b: Int) = a + b
    def zero = 0
  }

  implicit val DoubleMonoid = new Monoid[Double] {
    def add(a: Double, b: Double) = math.max(a, b)
    def zero = java.lang.Double.MIN_VALUE
  }

  implicit val StringMonoid = new Monoid[String] {
    def add(a: String, b: String) = a + b
    def zero = ""
  }
}

abstract class SegmentTreeNode[T] {
  var leftChild: SegmentTreeNode[T]
  var rightChild: SegmentTreeNode[T]
  var leftIdx: Int
  var rightIdx: Int
  var partialSum: T
}

class SegmentTree[T: Monoid] (arr: Array[T]) {
  // the SegmentTree should work for any Monoid
  val m = implicitly[Monoid[T]]
  var root = buildTree(arr, 0, arr.length-1)

  def findMiddleIdx(lIdx: Int, rIdx: Int): Int = {
    val mIdx: Int = (lIdx + rIdx) / 2
    return mIdx
  }

  def buildTree(arr: Array[T], lIdx: Int, rIdx: Int): SegmentTreeNode[T] = {
    if (lIdx == rIdx) {
      // leave nodes
      var node = new SegmentTreeNode[T] {
        var leftIdx = lIdx
        var rightIdx = rIdx
        var leftChild = null.asInstanceOf[SegmentTreeNode[T]]
        var rightChild = null.asInstanceOf[SegmentTreeNode[T]]
        var partialSum = arr(lIdx)
      }
      return node
    } else {
      // recursively build tree
      val mIdx: Int = findMiddleIdx(lIdx, rIdx)
      var node = new SegmentTreeNode[T] {
        var leftIdx = lIdx
        var rightIdx = rIdx
        var leftChild = buildTree(arr, lIdx, mIdx)
        var rightChild = buildTree(arr, mIdx+1, rIdx)
        var partialSum = m.zero
      }
      node.partialSum = m.add(node.leftChild.partialSum,
                              node.rightChild.partialSum)
      return node
    }
  }

  def getSum(node: SegmentTreeNode[T], l: Int, r: Int): T = {
    // Abstract partial sum
    if (node.leftIdx >= l && node.rightIdx <= r) {
      return node.partialSum
    } else if (node.rightIdx < l || node.leftIdx > r) {
      return m.zero
    } else {
      return m.add(getSum(node.leftChild, l, r), getSum(node.rightChild, l, r))
    }
  }

  def update(node: SegmentTreeNode[T], idx: Int, value: T): Unit = {
    if((node.leftIdx == idx) && (node.rightIdx == idx)) {
      node.partialSum = value
    } else if (node.leftIdx > idx || node.rightIdx < idx) {
      // skip
    } else {
      // propogate updated partialSum from leaves
      val mIdx: Int = findMiddleIdx(node.leftIdx, node.rightIdx)
      if(idx > mIdx) {
        update(node.rightChild, idx, value)
      } else {
        update(node.leftChild, idx, value)
      }

      node.partialSum = m.add(node.leftChild.partialSum, node.rightChild.partialSum)
    }
  }
}

object TestSegmentTree {
  def main(args: Array[String]): Unit = {
    // Test Integer Array
    val intArray = Array(1,4,3,2,1,1,0,5,10) //.map((x: Int) => x.toString)
    var intSegTree = new SegmentTree(intArray)

    println(intSegTree.getSum(intSegTree.root, 1, 4)) // 10
    intSegTree.update(intSegTree.root, 3, 7)
    println(intSegTree.getSum(intSegTree.root, 1, 4)) // 15

    // Test String Array
    val strArray = Array(1,4,3,2,1,1,0,5,10).map((x: Int) => x.toString)
    var strSegTree = new SegmentTree(strArray)

    println(strSegTree.getSum(strSegTree.root, 1, 4)) // "4321"
    strSegTree.update(strSegTree.root, 3, "7")
    println(strSegTree.getSum(strSegTree.root, 1, 4)) // "4371"

    // Test Partial Max
    val doubleArray = Array(1.0, 4.2, 3.5, 2.2, 1.8, 1.8, 0, 5, 10)
    var doubleSegTree = new SegmentTree(doubleArray)

    println(doubleSegTree.getSum(doubleSegTree.root, 1, 4)) // 4.2
    doubleSegTree.update(doubleSegTree.root, 3, 7.7)
    println(doubleSegTree.getSum(doubleSegTree.root, 1, 4)) // 7.7
  }
}
