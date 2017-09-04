import Layer.{FirstLayer, HigherLayer, Process}
import Master.{Exit, Start}
import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import breeze.linalg.DenseVector

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._

/**
  * Created by LD on 2017-05-24.
  */
object Master {

  case object Start

  case object Exit
}

class Master(training_data: Seq[(DenseVector[Double], DenseVector[Double])],
             epochs: Int,
             mini_batch_size: Int,
             eta: Double,
             test_data: Seq[(DenseVector[Double], DenseVector[Double])],
             sizes: Seq[Int]) extends Actor with ActorLogging{

  var start: Long = _

  val numberOfLayers: Int = sizes.length - 1


  //create layer actors for this shard's model replica
  val layers: Array[ActorRef] = new Array[ActorRef](numberOfLayers)

  for(l <- 0 until numberOfLayers) {
    layers(l) = context.actorOf(Props(new Layer(
      layerId = l,
      lowerLayer = if(l > 0) Some(layers(l - 1)) else None,
      eta = eta,
      sizes = sizes
    )), s"Layer-$l")

    if(l > 0) layers(l - 1) ! HigherLayer(layers(l))
    if(l == numberOfLayers - 1) layers(l) ! FirstLayer(layers.head)
  }

  override def receive: Receive = {
    case Start =>

      start = System.currentTimeMillis()
      log.info("Start!")
      layers.head ! Process(training_data, test_data, epochs, mini_batch_size)

    case Exit =>
      log.info(s"Total Time: ${System.currentTimeMillis() - start}ms")

      context.system.scheduler.scheduleOnce(2 seconds) {
        context.system.terminate()
      }
  }
}
