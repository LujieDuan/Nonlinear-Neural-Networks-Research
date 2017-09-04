import Common._
import Model._
import ParameterServer.{Result, Update}
import akka.actor.{Actor, ActorRef}
import breeze.linalg.{DenseMatrix, DenseVector, argmax, sum}

/**
  * Created by LD on 2017-05-24.
  */
object Model {

  case class Process(dataset: SetsType, epochs: Int, batchSize: Int)

  case class DoneTraining(replicaId: Int)

  case class TestResult(replicaId: Int, stats: StatType, costs: CostType)
}

class Model(replicaId: Int,
            parameterServer: ActorRef,
            eta: Float,
            sizes: Seq[Int]) extends Actor{

  var training_data: Iterator[Seq[(DenseVector[Float], DenseVector[Float])]] = _

  var datasets: SetsType = _

  val layers = sizes.length - 1

  def receive = {
    case Process(sets, epochs, size) => {
      training_data = sets._1.grouped(size)
      datasets = sets
      context.become(ReadyToProcess)
    }
  }

  def ReadyToProcess: Receive = {
    case Result(weights, biases) => {
      if (!training_data.hasNext){
        sender() ! DoneTraining(replicaId)
        context.become(ReadyToCheck)
      } else {
        val r = update_mini_batch(training_data.next(), weights, biases)
        sender() ! Update(r._1, r._2)
      }
    }
  }

  def ReadyToCheck: Receive = {
    case Result(weights, biases) => {
      val training_result_vector = datasets._1.map(x => (feedForward(x._1, weights, biases), x._2))
      val training_cost = computeLoss(training_result_vector)
      val training_result = training_result_vector.map(x => (argmax(x._1), argmax(x._2)))

      val validation_result_vector = datasets._2.map(x => (feedForward(x._1, weights, biases), x._2))
      val validation_cost = computeLoss(validation_result_vector)
      val validation_result = validation_result_vector.map(x => (argmax(x._1), argmax(x._2)))

      val test_result_vector = datasets._3.map(x => (feedForward(x._1, weights, biases), x._2))
      val test_cost = computeLoss(test_result_vector)
      val test_result = test_result_vector.map(x => (argmax(x._1), argmax(x._2)))

      val stat = new StatType{
        trainTotal = datasets._1.length
        trainPassed = training_result.toArray.count(x => x._1 == x._2)
        validationTotal = datasets._2.length
        validationPassed = validation_result.toArray.count(x => x._1 == x._2)
        testTotal = datasets._3.length
        testPassed = test_result.toArray.count(x => x._1 == x._2)
      }
      val cost = new CostType{
        trainTotal = training_cost
        validationTotal = validation_cost
        testTotal = test_cost
      }
      sender() ! TestResult(replicaId, stat, cost)
      context.unbecome()
    }
  }

  def update_mini_batch(mini_batch: Seq[(DenseVector[Float], DenseVector[Float])], weights: Seq[DenseMatrix[Seq[(Float, Float)]]],
                        biases: Seq[DenseVector[Float]]): (Seq[DenseMatrix[Seq[(Float, Float)]]], Seq[DenseVector[Float]]) = {
    var nable_b = biases.map(x => DenseVector.zeros[Float](x.length))
    var nable_w = cloneWeights(weights)
    mini_batch.foreach(x => {
      val delta = backprop(x._1, x._2, weights, biases)
      nable_b = (nable_b, delta._1).zipped.map((nb, dnb) => nb+dnb)
      nable_w = (nable_w, delta._2).zipped.map((nw, dnw) => addDelta(nw, dnw))
    })
    (nable_w.map(_.map(_.map(y => (y._1 * (eta/mini_batch.length), y._2 * (eta/mini_batch.length))))),
      nable_b.map(_.map(x => x * (eta/mini_batch.length))))

  }

  def feedForward(input: DenseVector[Float], weights: Seq[DenseMatrix[Seq[(Float, Float)]]], biases: Seq[DenseVector[Float]]): DenseVector[Float] = {
    var output :DenseVector[Float] = input
    (biases, weights).zipped.foreach((b, w) => output = sigmoid(DenseVector((0 until w.rows).map(x => sum(layerMultiply(w(x, ::).t, output))).toArray) + b))
    assert(output.length == sizes.last)
    output
  }

  def backprop(x: DenseVector[Float], y: DenseVector[Float], weights: Seq[DenseMatrix[Seq[(Float, Float)]]], biases: Seq[DenseVector[Float]]):
  (Seq[DenseVector[Float]], Seq[DenseMatrix[Seq[(Float, Float)]]]) = {
    var nable_b = biases.map(x => DenseVector.zeros[Float](x.length))
    var nable_w = cloneWeights(weights)
    var activation: DenseVector[Float] = x
    var activations: Seq[DenseVector[Float]] = Seq(activation)
    var zs: Seq[DenseVector[Float]] = Seq.empty
    (biases, weights).zipped.foreach((b, w) => {
      val z = DenseVector((0 until w.rows).map(x => sum(layerMultiply(w(x, ::).t, activation))).toArray) + b
      zs = zs :+ z
      activation = sigmoid(z)
      activations = activations :+ activation
    })
    //backward pass
    var delta = cost_derivative(activations(layers), y) * sigmoid_prime(zs(layers - 1))
    nable_b = nable_b.updated(layers - 1, delta)
    nable_w = nable_w.updated(layers - 1, transposeAndMultiply(activations(layers - 1), delta, weights(layers - 1)))

    //
    for(l <- 2 to layers) {
      val z = zs(layers - l)
      val sp = sigmoid_prime(z)
      val w = weights(layers - l + 1).t
      delta = DenseVector((0 until w.rows).map(x => sum(layerMultiplyPrime(w(x, ::).t, activations(layers - l + 1)(x)) * delta)).toArray) * sp
      nable_b = nable_b.updated(layers - l, delta)
      nable_w = nable_w.updated(layers - l, transposeAndMultiply(activations(layers - l), delta, weights(layers - l)))
    }
    (nable_b, nable_w)
  }

  def transposeAndMultiply(z: DenseVector[Float], t: DenseVector[Float],
                           w: DenseMatrix[Seq[(Float, Float)]]): DenseMatrix[Seq[(Float, Float)]] = {
    w.mapPairs((index, s) => s.zipWithIndex.map {
      case (v, i) =>
        if(i == s.length - 1) (t(index._1) * v._2 * pow(z(index._2), v._1) * lan(z(index._2)), t(index._1) * pow(z(index._2), v._1))
        else (0.0f, t(index._1) * pow(z(index._2), v._1))
    })
  }


  def layerMultiply(z: DenseVector[Seq[(Float, Float)]], t: DenseVector[Float]): DenseVector[Float] = {
    assert(z.length == t.length)
    z.mapPairs((index, zv) => sum(zv.map(zvv => zvv._2 * pow(t(index), zvv._1))))
  }

  def layerMultiplyPrime(z: DenseVector[Seq[(Float, Float)]], t: Float): DenseVector[Float] = {
    z.mapPairs((index, zv) => sum(zv.map(zvv => zvv._2 * zvv._1 * pow(t, zvv._1 - 1))))
  }

  def cloneWeights(z: Seq[DenseMatrix[Seq[(Float, Float)]]]): Seq[DenseMatrix[Seq[(Float, Float)]]] = {
    z.map(m => m.map(s => s.map(_ => (0.0f, 0.0f))))
  }

  def addDelta(z: DenseMatrix[Seq[(Float, Float)]], t: DenseMatrix[Seq[(Float, Float)]]): DenseMatrix[Seq[(Float, Float)]] = {
    assert(z.cols == t.cols && z.rows == t.rows)
    z.mapPairs((index, zv) => zv.zipAll(t(index._1, index._2), (0.0f, 0.0f), (0.0f, 0.0f)).map(x => (x._1._1 + x._2._1, x._1._2 + x._2._2)))
  }
}
