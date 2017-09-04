package actors

import Utils.{InputOutput, SimulationType}
import actors.DataShard.{FetchParameters, ReadyToProcess}
import actors.Layer.{ForwardPass, HigherLayer, LayerDoneFetch, ProcessBatch}
import actors.Master.ShardDone
import akka.actor.{Actor, ActorLogging, ActorRef, Props}

import scala.util.Random

/**
  * Created by LD on 2017-02-28.
  */
object DataShard {

  case object ReadyToProcess

  case object FetchParameters

}

class DataShard(shardId: Int,
                trainingData: Seq[InputOutput],
                parameterShard: Seq[Seq[Seq[ActorRef]]],
                simType: SimulationType.Value) extends Actor with ActorLogging{

  val miniBatchSize = 10

  val numberOfIterations = if (simType == SimulationType.Grid) Integer.MAX_VALUE else 10

  var iterationDone = 0

  // Number of layers without the input layer
  val numberOfLayers = parameterShard.size

  //parameter shard corresponding to each layer
  var trainingDataIterator = trainingData.grouped(miniBatchSize)

  //create layer actors for this shard's model replica
  val layers: Array[ActorRef] = new Array[ActorRef](numberOfLayers)

  for(l <- 0 until numberOfLayers) {
    layers(l) = context.actorOf(Props(new Layer(
      replicaId = shardId,
      layerId = l,
      lowerLayer = if(l > 0) Some(layers(l - 1)) else None,
      parameterShard = parameterShard(l),
      simType = simType
    )), s"Layer-$l")

    if(l > 0) layers(l - 1) ! HigherLayer(layers(l))
  }

  var layersToUpdate = (0 until numberOfLayers).toSet

  def receive = {
    case ReadyToProcess => {
      layers.foreach(_ ! FetchParameters)
      context.become(waitForFetchFinish)
    }
  }

  def waitForFetchFinish: Receive = {
    case LayerDoneFetch(layerId) => {
      layersToUpdate -= layerId
      if(layersToUpdate.isEmpty) {
        if(!trainingDataIterator.hasNext && numberOfIterations == iterationDone) {
          //log.info(s"Shard $shardId all done!")
          layers.foreach(_ ! FetchParameters)
          context.parent ! ShardDone(shardId)
          context.unbecome()
        }
        else{
          if (!trainingDataIterator.hasNext){
            //log.info(s"Shard $shardId done one iteration!")
            iterationDone += 1
            trainingDataIterator = Random.shuffle(trainingData).grouped(miniBatchSize)
          }
          val nextBatch = trainingDataIterator.next()
          layers.head ! ProcessBatch(nextBatch)
          //Wait the backward pass done so that can process the next input-output
          layersToUpdate = (0 until numberOfLayers).toSet
          context.unbecome()
        }
      }
    }
  }

}
