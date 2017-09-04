import breeze.linalg.DenseVector
import breeze.numerics.sigmoid

/**
  * Created by LD on 2017-05-24.
  */
object Compute {


  val Precision = 0.000001

  /**
    * Compare Doubles with precision
    */
  def ~=(x: Double, y: Double, precision: Double = Precision): Boolean = {
    if ((x - y).abs < precision) true else false
  }

  /**
    * Returns the value ln(i) when i is not 0. Otherwise return -Integer.MAX_VAL
    */
  def lan(i: Double) : Double = if (~=(i, 0)) -Integer.MAX_VALUE else scala.math.log(i)

  /**
    * Returns the por power of num
    */
  def pow(num: Double, por: Double): Double = {
    if (~=(por, 0) && ! ~=(num, 0))
      1
    else if (~=(por, 1.0))
      num
    else if (~=(num, 0))
      0
    else if (por < 0)
      1/scala.math.exp(lan(num)*(-por))
    else
      scala.math.exp(lan(num)*por)
  }

  /**
    * activation
    */
  def activation = (x: DenseVector[Double]) => x.map(el => sigmoid(el))

  /**
    * activation derivative
    */
  def activationDerivative = (x: DenseVector[Double]) => x.map(el => el * (1 - el))

}
