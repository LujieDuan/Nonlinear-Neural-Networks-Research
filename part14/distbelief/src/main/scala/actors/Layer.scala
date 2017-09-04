package actors

import Utils.Types.Delta
import Utils.Compute.{activation, activationDerivative}
import Utils.{InputOutput, SimulationType}
import actors.Checker.CheckNext
import actors.DataShard.{FetchParameters, ReadyToProcess}
import actors.Layer._
import actors.Synapse._
import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import breeze.linalg.{*, DenseMatrix, DenseVector, sum}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by LD on 2017-02-28.
  */
object Layer {

  case class ForwardPass(input: DenseVector[Double], output: DenseVector[Double])

  case class BackwardPass(deltas: DenseVector[Delta])

  case class LayerDoneFetch(layerId: Int)

  case class HigherLayer(higher: ActorRef)

  case class ProcessBatch(batch: Seq[InputOutput])

  val CONSTANT_ERROR_PRECISION = 0.4

}

class Layer(replicaId: Int,
            layerId: Int,
            lowerLayer: Option[ActorRef],
            parameterShard: Seq[Seq[ActorRef]],
            isChecker: Boolean = false,
            simType: SimulationType.Value) extends Actor with ActorLogging{

  val numberOfNeurons = parameterShard.size

  val numberOfInputs = parameterShard.head.size

  val numberOfSynapses = numberOfInputs * numberOfNeurons

  //log.info(s"Number of synapse: $numberOfSynapses $numberOfInputs $numberOfNeurons")

  val synapses = new Array[Array[ActorRef]](numberOfNeurons)

  var lastLayerInput: DenseVector[Double] = _

  var finalTarget: DenseVector[Double] = _

  for(i <- 0 until numberOfNeurons) {
    synapses(i) = new Array[ActorRef](numberOfInputs)
    for(j <- 0 until numberOfInputs) {
      synapses(i)(j) = context.actorOf(Props(new Synapse(
        replicaId = replicaId,
        layerId = layerId,
        inputId = j,
        outputId = i,
        synapseId = i * numberOfInputs + j,
        parameterShard = parameterShard(i)(j),
        layer = self
      )), s"Synapse-$j-$i")
    }
  }

  var synactsToUpdate = (0 until numberOfSynapses).toSet

  var higherLayer: Option[ActorRef] = None


  //Matrix of output:
  var outputMatrix = DenseMatrix.zeros[Double](numberOfInputs, numberOfNeurons)

  var outputToUpdate = (0 until numberOfSynapses).toSet

  //Propagations vector
  var propagationMatrix = DenseMatrix.zeros[Double](numberOfNeurons, numberOfInputs)

  var propagationToUpdate = (0 until numberOfSynapses).toSet

  var currentBatch: Iterator[InputOutput] = _


  def receive = {

    case HigherLayer(hl) => {
      higherLayer = Some(hl)
    }

    case FetchParameters => {
      synapses.foreach(_.foreach(_ ! FetchParameters))
      context.become(waitForFetchDone)
    }

    case ProcessBatch(batch) => {
      currentBatch = batch.toIterator
      val next = currentBatch.next()
      self ! ForwardPass(next.input, next.output)
    }

    case ForwardPass(input, output) => {
      lastLayerInput = input
      finalTarget = output
      synapses.foreach(_.zipWithIndex.foreach(x => x._1 ! SynapseForward(input(x._2))))
      context.become(processing)
    }

    case BackwardPass(deltas) => {
      synapses.zipWithIndex.foreach(x => x._1.foreach(_ ! SynapseBackward(deltas(x._2))))
      context.become(processing)
    }
  }

  def waitForFetchDone: Receive = {
    case SynapseDoneFetch(synapseId) => {
      synactsToUpdate -= synapseId

      //log.info(s"Synapse get parameters: layerId: $layerId, left: ${synactsToUpdate.size}.")
      if(synactsToUpdate.isEmpty) {

        //log.info(s"Layer get parameters: layerId: $layerId.")
        context.parent ! LayerDoneFetch(layerId)
        //Wait the backward pass done so that can process the next input-output
        synactsToUpdate = (0 until numberOfSynapses).toSet
        context.unbecome()
      }
    }
  }

  def processing: Receive = {

    case SynapseForwardDone(synapseId, inputId, outputId, result) => {

      outputMatrix(inputId, outputId) = result
      outputToUpdate -= synapseId
      if(outputToUpdate.isEmpty) {
        context.unbecome()
        outputToUpdate = (0 until numberOfSynapses).toSet

        val layerOutputs = sum(outputMatrix(::, *)).t
        val activatedOutputs = activation(layerOutputs)

        higherLayer match {

          //If there is a higher layer, continue forward
          case Some(hl) => hl ! ForwardPass(activatedOutputs, finalTarget)

          case _ => {
            //compute output
            val finalOutput = activatedOutputs
            isChecker match {
              case true => {
                //if this is the last layer, and this is a checker, check if result within error range
                context.unbecome()
                simType match {
                  case SimulationType.Grid => {
                    var errorSum = 0.0
                    breeze.linalg.zipValues(finalTarget, finalOutput).foreach((target, output) => errorSum += math.abs(target - output))
                    val continue = errorSum <= CONSTANT_ERROR_PRECISION
                    if(!continue)
                      log.info(s"Error: $errorSum")
                    context.parent ! CheckNext(finalOutput, continue)
                  }
                  case SimulationType.Mnist => {
                    var label =  finalTarget.toArray.indexOf(finalTarget.toArray.max)
                    var predicate =  finalOutput.toArray.indexOf(finalOutput.toArray.max)
                    val continue = label == predicate
                    if(!continue)
                      log.info(s"Error: $label vs $predicate")
                    label = 0
                    predicate = 0
                    context.parent ! CheckNext(finalOutput, continue)
                  }
                }
              }
              case false => {
                //if this is the last layer, and this is not a checker, turn around and do backward propagation
                //compute delta
                //log.info("One Done!")
                val dow: ArrayBuffer[Double] = new ArrayBuffer[Double]
                breeze.linalg.zipValues(finalTarget, finalOutput).foreach((target, output) => dow += (output - target) * output * (1 - output))
                //for synapses in this layer, do backward propagation first
                self ! BackwardPass(DenseVector(dow.toArray))
              }
            }
          }
        }
      }
    }

    case SynapseBackwardDone(synapseId, inputId, outputId, prevsum) => {
      propagationMatrix(outputId, inputId) = prevsum
      propagationToUpdate -= synapseId
      if(propagationToUpdate.isEmpty) {
        context.unbecome()
        propagationToUpdate = (0 until numberOfSynapses).toSet

        val propagationVector = sum(propagationMatrix(::, *)).t
        val activatedPropagation = activationDerivative(lastLayerInput) *:* propagationVector //elements-wise multiplication
        lowerLayer match {

          //If there is lower layer, continue
          case Some(ll) => ll ! BackwardPass(activatedPropagation)

          //All done, tell the data shard to learn the next input-output
          case _ =>
            if(currentBatch.hasNext){
              val next = currentBatch.next()
              self ! ForwardPass(next.input, next.output)
            }
            else
              context.parent ! ReadyToProcess
        }
      }
    }
  }
}
