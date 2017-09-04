package actors

import Utils.Compute.{lan, pr}
import Utils.Types.Coeff
import actors.DataShard.FetchParameters
import actors.ParameterShard.{LatestParameter, LatestParameterWithNeuronConstant, ParameterRequest, ParameterRequestWithNeuronConstant}
import actors.Synapse._
import akka.actor.{Actor, ActorLogging, ActorRef}
import breeze.linalg.{DenseVector, sum}

/**
  * Created by LD on 2017-03-08.
  */
object Synapse {

  case class SynapseDoneFetch(synapseId: Int)

  case class SynapseUpdateParameter(coeffChange: Coeff, expChange: Double)

  case class SynapseUpdateParameterWithNeuronConstant(coeffChange: Coeff, expChange: Double, neuronConstantChange: Double)

  case class SynapseForward(input: Double)

  case class SynapseForwardDone(synapseId: Int, inputId: Int, outputId: Int, output: Double)

  case class SynapseBackward(propagation: Double)

  case class SynapseBackwardDone(synapseId: Int, inputId: Int, outputId: Int, sum: Double)
}

class Synapse(replicaId: Int,
              layerId: Int,
              inputId: Int,
              outputId: Int,
              synapseId: Int,
              parameterShard: ActorRef,
              layer: ActorRef) extends Actor with ActorLogging{

  var latestCoeff: Coeff = _

  var latestExpo: Double = _

  var latestNeuronConstant: Double = _

  var firstInputOfNeuron: Boolean = inputId == 0

  var latestInputs: Double = _

  var batchCoeffChange: Seq[DenseVector[Double]] = Seq.empty

  var batchExpChange: Double = 0.0

  var batchConstantChange: Double = 0.0

  def receive = {

    case FetchParameters => {
      val batchSize = batchCoeffChange.size
      if(batchSize > 0) {
        val averageCoeffChange = batchCoeffChange.foldLeft(DenseVector.zeros[Double](batchCoeffChange.head.length))(_ + _).map(x => x / batchSize)
        firstInputOfNeuron match {
          case true => parameterShard ! SynapseUpdateParameterWithNeuronConstant(averageCoeffChange, batchExpChange / batchSize, batchConstantChange / batchSize)
          case false => parameterShard ! SynapseUpdateParameter(averageCoeffChange, batchExpChange / batchSize)
        }
        batchCoeffChange = Seq.empty
        batchExpChange = 0.0
        batchConstantChange = 0.0
      }

      firstInputOfNeuron match {
        case true => parameterShard ! ParameterRequestWithNeuronConstant(replicaId)
        case false => parameterShard ! ParameterRequest(replicaId)
      }
      context.become(waitForParameters)
    }

    case SynapseForward(input) => {
      //log.info(s"Synapse Forward: layerId: $layerId, inputId: $inputId, outputId: $outputId, input $input.")
      latestInputs = input
      val polynomialTerms: DenseVector[Double] = DenseVector(((1 until latestCoeff.length).map(x => pr(input, x)) :+ pr(input, latestExpo)).reverse.toArray)
      val polynomialResult: Double = sum(polynomialTerms *:* latestCoeff)
      val result: Double = firstInputOfNeuron match {
        case true => polynomialResult + latestNeuronConstant
        case false => polynomialResult
      }
      layer ! SynapseForwardDone(synapseId, inputId, outputId, result)
    }

    case SynapseBackward(deltas) => {
      //log.info(s"Synapse backward: layerId: $layerId, inputId: $inputId, outputId: $outputId, input $deltas.")
      //First, compute the deltas/changes
      val coeffChange = DenseVector(((1 until latestCoeff.length).map(x => deltas * pr(latestInputs, x)) :+ (deltas * pr(latestInputs, latestExpo))).reverse.toArray)
      val expChange = deltas * latestCoeff(0) * pr(latestInputs, latestExpo) * lan(latestInputs)
      val neuronConstantChange: Double = deltas
      batchCoeffChange = batchCoeffChange :+ coeffChange
      batchExpChange += expChange
      batchConstantChange += neuronConstantChange

      //Second, compute the sum/dow values for lower layer to propagate
      val expVector = DenseVector(((1 until latestCoeff.length).map(x => x * pr(latestInputs, x - 1)) :+ (latestExpo * pr(latestInputs, latestExpo - 1))).reverse.toArray)
      val timesCoeff = expVector *:* latestCoeff
      val prevsum: Double = deltas * sum(timesCoeff)

      layer ! SynapseBackwardDone(synapseId, inputId, outputId, prevsum)
    }

  }

  def waitForParameters: Receive = {
    case LatestParameter(c, e) => {
      //log.info(s"Synapse get parameters: layerId: $layerId, inputId: $inputId, outputId: $outputId.")

      latestExpo = e
      latestCoeff = c
      layer ! SynapseDoneFetch(synapseId)
      context.unbecome()
    }

    case LatestParameterWithNeuronConstant(c, e, constant) => {
      //log.info(s"Synapse get parameters: layerId: $layerId, inputId: $inputId, outputId: $outputId.")

      latestNeuronConstant = constant
      latestExpo = e
      latestCoeff = c
      layer ! SynapseDoneFetch(synapseId)
      context.unbecome()
    }
  }

}
