import Common._
import Master.{Exit, Start}
import Model.Process
import ParameterServer.{PackParameter, Replicas}
import akka.actor.{Actor, ActorLogging, ActorRef, Props}

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._

/**
  * Created by LD on 2017-05-24.
  */
object Master {

  case object Start

  case class Exit(stats: StatType, costs: CostType)
}

class Master(datasets: SetsType,
             epochs: Int,
             mini_batch_size: Int,
             learningRate: Float,
             sizes: Seq[Int],
             replicaCount: Int) extends Actor with ActorLogging{

  var start: Long = _

  implicit var fileName = ""

  val numberOfLayers: Int = sizes.length - 1

  val parameterServer: ActorRef = context.actorOf(Props(new ParameterServer(
    sizes = sizes,
    replicaCount = replicaCount
  )))

  //create layer actors for this shard's model replica
  val models: Array[ActorRef] = new Array[ActorRef](replicaCount)

  var costSets: CostsType = Common.initialCosts()

  for(l <- 0 until replicaCount) {
    models(l) = context.actorOf(Props(new Model(
      replicaId = l,
      parameterServer = parameterServer,
      eta = learningRate,
      sizes = sizes
    )), s"Model-$l")
  }

  var iterationDone: Int = 0

  var bestValidationAccuracy = 0.0f

  override def receive: Receive = {
    case Start =>

      start = System.currentTimeMillis()
      fileName = Common.generateFileName(learningRate, mini_batch_size, start, sizes, "ActorNN")
      Common.log(s"Actor Based Nonlinear Neural Nets:${sizes.mkString("-")}")
      Common.log(s"Batch Size: $mini_batch_size. Learning Rate: $learningRate. Iterations: $epochs. Replica Counts: $replicaCount.")
      val training = datasets._1.grouped(datasets._1.length/replicaCount + 1)
      val validation = datasets._2.grouped(datasets._2.length/replicaCount + 1)
      val testing = datasets._3.grouped(datasets._3.length/replicaCount + 1)
      models.foreach(_ ! Process((training.next(), validation.next(), testing.next()), epochs, mini_batch_size))
      parameterServer ! Replicas(models)

    case Exit(stats, costs) =>
      iterationDone += 1
      costSets._1 += ((System.currentTimeMillis() - start, costs.trainTotal))
      costSets._2 += ((System.currentTimeMillis() - start, costs.validationTotal))
      costSets._3 += ((System.currentTimeMillis() - start, costs.testTotal))

      Common.log(s"Epoch $iterationDone: Train Set:     ${stats.trainPassed.toFloat / stats.trainTotal}")
      Common.log(s"         Validation Set:     ${stats.validationPassed.toFloat / stats.validationTotal}")
      Common.log(s"         Test Set:     ${stats.testPassed.toFloat / stats.testTotal}")
      val validationAccuracy = stats.validationPassed.toFloat / stats.validationTotal
      if (validationAccuracy > bestValidationAccuracy) bestValidationAccuracy = validationAccuracy

      if(iterationDone < epochs) {
        val training = datasets._1.grouped(datasets._1.length/replicaCount + 1)
        val validation = datasets._2.grouped(datasets._2.length/replicaCount + 1)
        val testing = datasets._3.grouped(datasets._3.length/replicaCount + 1)
        models.foreach(_ ! Process((training.next(), validation.next(), testing.next()), epochs, mini_batch_size))
        parameterServer ! Replicas(models)
      } else {
        summaryLog(s"$bestValidationAccuracy $fileName")
        Common.outputResults(datasets, costSets, start)
        parameterServer ! PackParameter(fileName)
        context.system.scheduler.scheduleOnce(2 seconds) {
          context.system.terminate()
        }
      }
  }
}
