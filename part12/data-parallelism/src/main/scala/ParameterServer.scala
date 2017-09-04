import Common.{CostType, StatType}
import Master.Exit
import Model.{DoneTraining, TestResult}
import Pack.packWeightsExp
import ParameterServer._
import akka.actor.{Actor, ActorLogging, ActorRef}
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.Rand

/**
  * Created by LD on 2017-05-24.
  */
object ParameterServer {
  case class Update(weights: Seq[DenseMatrix[Seq[(Float, Float)]]], biases:  Seq[DenseVector[Float]])

  case class Result(weights: Seq[DenseMatrix[Seq[(Float, Float)]]], biases:  Seq[DenseVector[Float]])

  case object Output

  case class Replicas(replicas: Seq[ActorRef])

  case class PackParameter(fileName: String)
}

class ParameterServer(sizes: Seq[Int], replicaCount: Int) extends Actor with ActorLogging{

  var latestWeights: Seq[DenseMatrix[Seq[(Float, Float)]]] = sizes.drop(1).zipWithIndex.map(x => DenseMatrix.rand(x._1, sizes(x._2), rand = Rand.gaussian)).map(x => x.map(y => Seq[(Float, Float)]((1.0f, y.toFloat))))

  var latestNeuronConstant: Seq[DenseVector[Float]] = sizes.drop(1).map(x => DenseVector.rand(x, rand = Rand.gaussian).map(_.toFloat))

  var replicas: Set[Int] = (0 until replicaCount).toSet

  var replicasRef: Seq[ActorRef] = _

  var statTotal: StatType = new StatType

  var epoch: Int = 0

  var costsTotal: CostType = new CostType

  def receive = {

    case Update(nable_w, nable_b) => {

      latestNeuronConstant = (latestNeuronConstant, nable_b).zipped.map((b, nb) => b - nb)
      latestWeights = (latestWeights, nable_w).zipped.map((w, nw) => minusDelta(w, nw))
      latestWeights = addExponential(latestWeights)

      context.sender() ! Result(latestWeights, latestNeuronConstant)
    }

    case DoneTraining(replicaId) => {
      replicas -= replicaId
      if (replicas.isEmpty) {
        replicasRef.foreach(_ ! Result(latestWeights, latestNeuronConstant))
        replicas = (0 until replicaCount).toSet
      }
    }

    case TestResult(replicaId, stats, costs) => {
      statTotal.add(stats)
      costsTotal.add(costs)
      replicas -= replicaId
      if (replicas.isEmpty) {
        context.parent ! Exit(statTotal, costsTotal)
        epoch += 1
        replicas = (0 until replicaCount).toSet
      }
    }

    case Replicas(rs) => {
      replicasRef = rs
      statTotal = new StatType
      costsTotal = new CostType
      replicasRef.foreach(_ ! Result(latestWeights, latestNeuronConstant))
    }

    case Output => {
      latestWeights.zipWithIndex.foreach(ma => ma._1.foreachValue(se => log.info(s"Layer: ${ma._2 + 1} ${se.map(t => t._2.toString + "x^"+ t._1.toString + " ").mkString}")))
    }

    case PackParameter(fileName) => {
      packWeightsExp(latestNeuronConstant, latestWeights, fileName)
    }
  }

  def minusDelta(z: DenseMatrix[Seq[(Float, Float)]], t: DenseMatrix[Seq[(Float, Float)]]): DenseMatrix[Seq[(Float, Float)]] = {
    assert(z.cols == t.cols && z.rows == t.rows)
    z.mapPairs((index, zv) => zv.zipAll(t(index._1, index._2), (0.0f, 0.0f), (0.0f, 0.0f)).map(x => {
      if(x._1._1 - x._2._1 < 0) (0.0f, x._1._2 - x._2._2)else (x._1._1 - x._2._1, x._1._2 - x._2._2)
    }))
  }

  def addExponential(z: Seq[DenseMatrix[Seq[(Float, Float)]]]): Seq[DenseMatrix[Seq[(Float, Float)]]] = {
    z.map(w => w.map(s => {
      if(s.last._1 > s.length + 1) {
        val original_length = s.length
        val original_exp = s.last._1
        val original_coeff = s.last._2
        val updated = s.updated(original_length - 1, (original_length.toFloat, original_coeff))
        updated :+ (original_exp, 0.0f)
      } else s
    }))
  }
}
