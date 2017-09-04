import breeze.linalg.{DenseMatrix, DenseVector, argmax, reshape, sum}
import breeze.plot._

import scala.collection.mutable.ArrayBuffer

/**
  * Created by LD on 2017-07-10.
  * Common Functions
  */
class DataSet(train: Seq[(DenseVector[Float], DenseVector[Float])],
              validation: Seq[(DenseVector[Float], DenseVector[Float])],
              test: Seq[(DenseVector[Float], DenseVector[Float])]) {

  val Train: Seq[(DenseVector[Float], DenseVector[Float])] = train
  val Validation: Seq[(DenseVector[Float], DenseVector[Float])] = validation
  val Test: Seq[(DenseVector[Float], DenseVector[Float])] = test
}

class CostSet(train: ArrayBuffer[(Long, Float)],
              validation: ArrayBuffer[(Long, Float)],
              test: ArrayBuffer[(Long, Float)]){

  val Train: ArrayBuffer[(Long, Float)] = train
  val Validation: ArrayBuffer[(Long, Float)] = validation
  val Test: ArrayBuffer[(Long, Float)] = test
}

object Common {

  val Precision = 0.00000000000000000000001f

  /**
    * Compare Floats with precision
    */
  def ~=(x: Float, y: Float, precision: Float = Precision): Boolean = {
    if ((x - y).abs < precision) true else false
  }

  /**
    * Returns the value ln(i) when i is not 0. Otherwise return -Integer.MAX_VAL
    */
  def lan(i: Float) : Float = if (~=(i, 0)) -Integer.MAX_VALUE else scala.math.log(i).toFloat

  /**
    * Returns the por power of num
    */
  def pow(num: Float, por: Float): Float = {
    assert(num >= 0)
    if (~=(por, 0) && ! ~=(num, 0))
      1
    else if (~=(por, 1.0f))
      num
    else if (~=(num, 0))
      0
    else if (por < 0)
      1/scala.math.exp(lan(num)*(-por)).toFloat
    else
      scala.math.exp(lan(num)*por).toFloat
  }

  /**
    * Returns the por power of num
    * @param num
    * @param por
    * @return
    */
  def powVector(num: DenseVector[Float], por: DenseVector[Float]): DenseVector[Float] = {
    por.mapPairs((i, p) => pow(num(i),p))
  }

  /**
    * Returns the por power of num
    * @param num
    * @param por
    * @return
    */
  def pow2(num: Float, por: DenseVector[Float]): DenseVector[Float] = {
    por.mapValues(p => pow(num,p))
  }

  /**
    *
    * @param output_activation
    * @param y
    * @return
    */
  def cost_derivative(output_activation: DenseVector[Float], y: DenseVector[Float]): DenseVector[Float] = {
    output_activation - y
  }

  /**
    *
    * @param z
    * @return
    */
  def sigmoid(z: DenseVector[Float]): DenseVector[Float] = {
    z.map(x => breeze.numerics.sigmoid(x))
  }

  /**
    *
    * @param z
    * @return
    */
  def sigmoid_prime(z: DenseVector[Float]): DenseVector[Float] = {
    z.map(el => breeze.numerics.sigmoid(el) * (1 - breeze.numerics.sigmoid(el)))
  }

  /**
    *
    * @param results
    * @return
    */
  def computeLoss(results: Seq[(DenseVector[Float], DenseVector[Float])]): Float = {
    val costs = results.map(x => sum((x._2 - x._1).map(y => math.pow(y, 2).toFloat)))
    costs.foldLeft(0.0f)(_ + _)
  }

  /**
    *
    * @param test_data
    * @param forwardFunc
    * @param startTime
    * @return
    */
  def evaluate(test_data: Seq[(DenseVector[Float], DenseVector[Float])], forwardFunc: (DenseVector[Float]) => DenseVector[Float], startTime: Long): (Int, (Long, Float)) = {
    val test_result_vector = test_data.map(x => (forwardFunc(x._1), x._2))
    val cost = (System.currentTimeMillis() - startTime, computeLoss(test_result_vector))
    val test_result = test_result_vector.map(x => (argmax(x._1), argmax(x._2)))
    (test_result.toArray.count(x => x._1 == x._2), cost)
  }

  /**
    *
    * @param ws
    */
  private def plotWeights(ws: Seq[DenseMatrix[Float]]) = {
    val f2 = Figure()
    f2.subplot(0) += image(reshape(ws(0)(2, ::), 28, 28).map(_.toDouble))
    f2.saveas("image.png")
  }

  /**
    *
    * @param dataset
    * @param costs
    * @param epochCount
    * @param startTime
    * @param forwardFunc
    */
  def evaluteSets(dataset: DataSet, costs: CostSet,
                  epochCount: Int, startTime: Long, forwardFunc: (DenseVector[Float]) => DenseVector[Float]) = {
    val trainResults = evaluate(dataset.Train, forwardFunc, startTime)
    val validationResults = evaluate(dataset.Validation, forwardFunc, startTime)
    val testResults = evaluate(dataset.Test, forwardFunc, startTime)

    costs.Train += trainResults._2
    costs.Validation += validationResults._2
    costs.Test += testResults._2

    println(s"Epoch $epochCount: Train Set: ${trainResults._1} / ${dataset.Train.length} ${ 100.0 * trainResults._1 / dataset.Train.length}%")
    println(s"         Validation Set: ${validationResults._1} / ${dataset.Validation.length} ${ 100.0 * validationResults._1 / dataset.Validation.length}%")
    println(s"         Test Set: ${testResults._1} / ${dataset.Test.length} ${ 100.0 * testResults._1 / dataset.Test.length}%")
  }

  /**
    *
    * @return
    */
  def initialCosts(): CostSet = {
    val trainCosts: ArrayBuffer[(Long, Float)] = new ArrayBuffer[(Long, Float)]()

    val validationCosts: ArrayBuffer[(Long, Float)] = new ArrayBuffer[(Long, Float)]()

    val testCosts: ArrayBuffer[(Long, Float)] = new ArrayBuffer[(Long, Float)]()

    new CostSet(trainCosts, validationCosts, testCosts)
  }

  /**
    *
    * @param dataset
    * @param costs
    * @param epochs
    * @param mini_batch_size
    * @param eta
    * @param startTime
    * @param updateFunc
    * @param forwardFunc
    */
  def SGD(dataset: DataSet, costs: CostSet,
          epochs: Int, mini_batch_size: Int, eta: Float, startTime: Long,
          updateFunc: (Seq[(DenseVector[Float], DenseVector[Float])], Float) => Unit,
          forwardFunc: (DenseVector[Float]) => DenseVector[Float]) = {
    var mini_batches: Iterator[Seq[(DenseVector[Float], DenseVector[Float])]] = Iterator.empty
    for(j <- 0 until epochs) {

      //Switch the next two lines to random shuffle the dataset before each iteration/epoch
      //mini_batches = scala.util.Random.shuffle(dataset.Train).grouped(mini_batch_size)
      mini_batches = dataset.Train.grouped(mini_batch_size)

      mini_batches.foreach(x => updateFunc(x, eta))

      evaluteSets(dataset, costs, j, startTime, forwardFunc)
    }
  }
}
