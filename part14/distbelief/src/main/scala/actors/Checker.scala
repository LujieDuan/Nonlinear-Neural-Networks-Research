package actors

import Utils.{InputOutput, SimulationType}
import actors.Checker._
import actors.DataShard.{FetchParameters, ReadyToProcess}
import actors.Layer.{ForwardPass, HigherLayer, LayerDoneFetch}
import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import breeze.linalg.DenseVector

import scala.collection.mutable.ArrayBuffer

/**
  * Created by LD on 2017-03-15.
  */
object Checker {

  case class CheckerDone(checkerId: Int, pass: Boolean, positive: Int)

  case class CheckNext(result: DenseVector[Double], continue: Boolean)

  case class GatherOutput(trainingData: Seq[InputOutput])

  case class OutputDone(checkerId: Int, output: Array[InputOutput])
}

class Checker(checkId: Int,
              trainingData: Seq[InputOutput],
              parameterShard: Seq[Seq[Seq[ActorRef]]],
              simType: SimulationType.Value) extends Actor with ActorLogging{

  // Number of layers without the input layer
  val numberOfLayers = parameterShard.size

  //parameter shard corresponding to each layer
  var trainingDataIterator = trainingData.toIterator

  //create layer actors for this shard's model replica
  val layers: Array[ActorRef] = new Array[ActorRef](numberOfLayers)

  for(l <- 0 until numberOfLayers) {
    layers(l) = context.actorOf(Props(new Layer(
      replicaId = checkId,
      layerId = l,
      lowerLayer = if(l > 0) Some(layers(l - 1)) else None,
      parameterShard = parameterShard(l),
      isChecker = true,
      simType = simType
    )), s"Layer-$l")

    if(l > 0) layers(l - 1) ! HigherLayer(layers(l))
  }

  var layersToUpdate = (0 until numberOfLayers).toSet

  val outputArray: ArrayBuffer[InputOutput] = ArrayBuffer[InputOutput]()

  var nextInputOutputToGather: InputOutput = _

  var passed = 0

  def receive = {
    case ReadyToProcess => {
      //log.info("READY")
      trainingDataIterator = trainingData.toIterator
      layers.foreach(_ ! FetchParameters)
      context.become(waitForFetchFinish)
    }
    case CheckNext(output, continue) => {
      simType match{
        case SimulationType.Grid => {
          //For Grid problem, if any sample is not convergent, then start again
          if(!continue) {
            //start over
            //log.info("ABORT")
            context.parent ! CheckerDone(checkId, false, 0)
          }
          else if(trainingDataIterator.hasNext) {
            //log.info("Check")
            val nextInputOutput = trainingDataIterator.next()
            layers.head ! ForwardPass(nextInputOutput.input, nextInputOutput.output)
          }
          else {
            context.parent ! CheckerDone(checkId, true, 0)
          }
        }
        case SimulationType.Mnist => {
          //For Mnist problem, only go through the set one time to test for error ratio
          if(continue) passed += 1
          if(trainingDataIterator.hasNext) {
            //log.info("Check")
            val nextInputOutput = trainingDataIterator.next()
            layers.head ! ForwardPass(nextInputOutput.input, nextInputOutput.output)
          }
          else {
            context.parent ! CheckerDone(checkId, true, passed)
          }
        }
      }
    }
    case GatherOutput(dataSet) => {
      trainingDataIterator = dataSet.toIterator
      context.become(gatherOutput)
      self ! CheckNext(DenseVector(0), true)
    }
  }

  def waitForFetchFinish: Receive = {
    case LayerDoneFetch(layerId) => {
      layersToUpdate -= layerId
      if(layersToUpdate.isEmpty) {
        //Wait the backward pass done so that can process the next input-output
        layersToUpdate = (0 until numberOfLayers).toSet
        context.unbecome()
        self ! CheckNext(DenseVector(0), true)
      }
    }
  }

  def gatherOutput: Receive = {
    case CheckNext(output, continue) => {
      if (nextInputOutputToGather != null) outputArray += new InputOutput(nextInputOutputToGather.input, output)
      if (trainingDataIterator.hasNext) {
        nextInputOutputToGather = trainingDataIterator.next()
        layers.head ! ForwardPass(nextInputOutputToGather.input, nextInputOutputToGather.output)
      }
      else {
        context.parent ! OutputDone(checkId, outputArray.toArray)
        //context.stop(self)
      }
    }
  }
}
