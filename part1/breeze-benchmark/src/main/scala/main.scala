import breeze.linalg.DenseMatrix

/**
  * Created by LD on 2017-07-11.
  */
object main {

  def main(args: Array[String]): Unit = {

    println("Benchmark Started!")

//    BenchPow
//
//    TestPow

    benchMatrixElementwiseMultiplication

    benchMatrixApply

    benchMatrixMultiplication
  }


  private def BenchPow = {
    var time: Long = 0
    val step = 0.0001f

    time = System.currentTimeMillis()
    for (i <- 0.01f to 1000 by step) {
      Common.pow(i, i)
    }
    val powTime = System.currentTimeMillis() - time


    time = System.currentTimeMillis()
    for (i <- 0.01f to 1000 by step) {
      scala.math.pow(i, i)
    }
    val scalaTime = System.currentTimeMillis() - time


    time = System.currentTimeMillis()
    for (i <- 0.01f to 1000 by step) {
      breeze.numerics.pow(i, i)
    }
    val breezeTime = System.currentTimeMillis() - time


    println(s"My Function: $powTime, scala math: $scalaTime, Breeze: $breezeTime")
  }

  private def TestPow = {
    val step = 0.0001f

    var difference = 0.0
    for (i <- 0.01f to 1000 by step) {
      println(s"Diff: $difference")
      val x = scala.math.random()
      difference += math.abs(Common.pow(x, x) - scala.math.pow(x, x))
    }
  }

  private def benchMatrixMultiplication = {
    println("Testing matrix multiplication:")

    val time = System.currentTimeMillis()
    val matrix = DenseMatrix.rand(1000, 1000)
    for (i <- 0 until 1000) {
      val m = matrix * matrix
    }
    println(s"Total: ${(System.currentTimeMillis() - time) / 1000.0}s")
  }

  private def benchMatrixApply = {
    println("Testing matrix apply:")

    val time = System.currentTimeMillis()
    val matrix = DenseMatrix.rand(1000, 1000)
    for (i <- 0 until 1000) {
      val m = breeze.numerics.exp(matrix)
    }
    println(s"Total: ${(System.currentTimeMillis() - time) / 1000.0}s")
  }

  private def benchMatrixElementwiseMultiplication = {
    println("Testing matrix elementwise multiplication:")

    val time = System.currentTimeMillis()
    val matrix = DenseMatrix.rand(1000, 1000)
    for (i <- 0 until 1000) {
      val m = matrix *:* matrix
    }
    println(s"Total: ${(System.currentTimeMillis() - time) / 1000.0}s")
  }
}

object Common {

  val Precision = 0.00000000000000000000001f

  /**
    * Compare Doubles with precision
    */
  def ~=(x: Double, y: Double, precision: Double = Precision): Boolean = {
    if ((x - y).abs < precision) true else false
  }

  /**
    * Returns the value ln(i) when i is not 0. Otherwise return -Integer.MAX_VAL
    */
  def lan(i: Double): Double = if (~=(i, 0)) -Integer.MAX_VALUE else scala.math.log(i)

  /**
    * Returns the por power of num
    */
  def pow(num: Double, por: Double): Double = {
    assert(num >= 0)
    if (~=(por, 0) && ! ~=(num, 0))
      1
    else if (~=(por, 1.0))
      num
    else if (~=(num, 0))
      0
    else if (por < 0)
      1 / scala.math.exp(lan(num) * (-por))
    else
      scala.math.exp(lan(num) * por)
  }
}
