import java.io._
import java.sql.Timestamp

import Pack.packCosts
import breeze.linalg.{DenseMatrix, DenseVector, argmax, reshape, sum}
import breeze.plot._

import scala.collection.mutable.ArrayBuffer

/**
  * Created by LD on 2017-07-10.
  * Common Functions
  */
object Common {

  type CostsType = (ArrayBuffer[(Long, Float)], ArrayBuffer[(Long, Float)], ArrayBuffer[(Long, Float)])

  class StatType {
    var trainTotal = 0
    var trainPassed = 0
    var validationTotal = 0
    var validationPassed = 0
    var testTotal = 0
    var testPassed = 0

    def add(other: StatType) = {
      this.trainTotal += other.trainTotal
      this.trainPassed += other.trainPassed
      this.validationTotal += other.validationTotal
      this.validationPassed += other.validationPassed
      this.testTotal += other.testTotal
      this.testPassed += other.testPassed
    }
  }

  class CostType {
    var trainTotal = 0.0f
    var validationTotal = 0.0f
    var testTotal = 0.0f

    def add(other: CostType) = {
      this.trainTotal += other.trainTotal
      this.validationTotal += other.validationTotal
      this.testTotal += other.testTotal
    }
  }

  type SetsType = (Seq[(DenseVector[Float], DenseVector[Float])], Seq[(DenseVector[Float], DenseVector[Float])], Seq[(DenseVector[Float], DenseVector[Float])])

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
    * @param learning_rate
    * @param mini_batch_size
    * @param startTime
    * @param netStructure
    * @param filePrefix
    * @return
    */
  def generateFileName(learning_rate: Float, mini_batch_size: Int, startTime: Long,
                       netStructure: Seq[Int], filePrefix: String): String = {
    val timestamp = new Timestamp(startTime)
    val fileName = String.format(s"$filePrefix-${netStructure.mkString("-")}-batch-$mini_batch_size-rate-$learning_rate-$timestamp")
    fileName
  }

  /**
    *
    * @param dataset
    * @param costs
    * @param startTime
    * @param fileName
    */
  def outputResults(dataset: SetsType, costs: CostsType, startTime: Long)(implicit fileName: String): Unit = {
    if (System.getenv("PLOT_LOSTS") != "false") {
      val f = Figure(s"$fileName.png")
      val p = f.subplot(0)
      p += plot(costs._1.map(x => x._1 / 1000.0f), costs._1.map(x => x._2))
      p += plot(costs._2.map(x => x._1 / 1000.0f), costs._2.map(x => x._2), '+')
      p += plot(costs._3.map(x => x._1 / 1000.0f), costs._3.map(x => x._2), '.')
      p.xlabel = "time in s"
      p.ylabel = "cost"
      val directory = "plots/"
      val locationFile = new File(directory)

      if (!locationFile.exists)
        locationFile.mkdirs
      f.saveas(s"$directory$fileName.png")
    }
    packCosts(costs, fileName)
    log(s"Final costs on train-set: ${costs._1.last._2 / dataset._1.length} validation-set: ${costs._2.last._2 / dataset._2.length} " +
      s"testSet-set: ${costs._3.last._2 / dataset._3.length}")
    log(s"Total Time: ${System.currentTimeMillis() - startTime}ms")
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
  def evaluteSets(dataset: SetsType, costs: CostsType,
                  epochCount: Int, startTime: Long, forwardFunc: (DenseVector[Float]) => DenseVector[Float])(implicit fileName: String): Float = {
    val trainResults = evaluate(dataset._1, forwardFunc, startTime)
    val validationResults = evaluate(dataset._2, forwardFunc, startTime)
    val testResults = evaluate(dataset._3, forwardFunc, startTime)

    costs._1 += trainResults._2
    costs._2 += validationResults._2
    costs._3 += testResults._2

    log(s"Epoch $epochCount: Train Set:     ${trainResults._1.toFloat / dataset._1.length}")
    log(s"         Validation Set:     ${validationResults._1.toFloat / dataset._2.length}")
    log(s"         Test Set:     ${testResults._1.toFloat / dataset._3.length}")
    validationResults._1 / dataset._2.length
  }

  /**
    *
    * @return
    */
  def initialCosts(): CostsType = {
    val trainCosts: ArrayBuffer[(Long, Float)] = new ArrayBuffer[(Long, Float)]()

    val validationCosts: ArrayBuffer[(Long, Float)] = new ArrayBuffer[(Long, Float)]()

    val testCosts: ArrayBuffer[(Long, Float)] = new ArrayBuffer[(Long, Float)]()

    (trainCosts, validationCosts, testCosts)
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
  def SGD(dataset: SetsType, costs: CostsType,
          epochs: Int, mini_batch_size: Int, eta: Float, startTime: Long,
          updateFunc: (Seq[(DenseVector[Float], DenseVector[Float])], Float) => Unit,
          forwardFunc: (DenseVector[Float]) => DenseVector[Float])(implicit fileName: String) = {
    var mini_batches: Iterator[Seq[(DenseVector[Float], DenseVector[Float])]] = Iterator.empty
    var bestValidationError = Float.MaxValue
    for(j <- 0 until epochs) {
      mini_batches = scala.util.Random.shuffle(dataset._1).grouped(mini_batch_size)
      mini_batches.foreach(x => updateFunc(x, eta))

      val validationError = evaluteSets(dataset, costs, j, startTime, forwardFunc)
      if (validationError < bestValidationError) bestValidationError = validationError
      //plotWeights(weights)
    }
    summaryLog(s"$bestValidationError $fileName")
  }


  def log(str: String)(implicit fileName: String) = {
    println(str)
    val directory = "log/"
    val locationFile = new File(directory)

    if (!locationFile.exists)
      locationFile.mkdirs
    val pw = new PrintWriter(new FileOutputStream(new File(s"$directory$fileName.txt" ), true))
    pw.write(str)
    pw.write("\n")
    pw.close()
  }

  def summaryLog(str: String) = {
    val directory = "log/"
    val locationFile = new File(directory)

    if (!locationFile.exists)
      locationFile.mkdirs
    val pw = new PrintWriter(new FileOutputStream(new File(s"${directory}Summary.txt" ), true))
    pw.write(str)
    pw.write("\n")
    pw.close()
  }
}
