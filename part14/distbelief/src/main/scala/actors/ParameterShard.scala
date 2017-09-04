package actors

import Utils.Compute
import Utils.Types.Coeff
import actors.CheckerCoordinator.{ParametersCached, RefreshCachedParameters}
import actors.ParameterShard._
import actors.Synapse.{SynapseUpdateParameter, SynapseUpdateParameterWithNeuronConstant}
import akka.actor.Actor
import akka.actor.ActorLogging
import breeze.linalg.DenseVector

/**
  * Created by LD on 2017-02-28.
  */
object ParameterShard {

  case class ParameterRequest(dataShardId: Int)

  case class ParameterRequestWithNeuronConstant(dataShardId: Int)

  case class CachedParameterRequest(dataShardId: Int)

  case class CachedParameterRequestWithNeuronConstant(dataShardId: Int)

  case class LatestParameter(coeff: Coeff, exp: Double)

  case class LatestParameterWithNeuronConstant(coeff: Coeff, exp: Double, neuronConstant: Double)

  case object OutputCoefficients

  case object OutputCachedCoefficients

}

class ParameterShard(parameterShardId: Int,
                     layerId: Int,
                     inputId: Int,
                     outputId: Int,
                     lambdaC: Double,
                     lambdaE: Double,
                     initialCoeff: Coeff,
                     initialExpo: Double,
                     firstOfNeuron: Boolean) extends Actor with ActorLogging{

  var coeff: Coeff = initialCoeff

  var exponential: Double = initialExpo

  var neuronConstant: Double = Compute.random2

  var coeffCached: Coeff = coeff

  var exponentialCached: Double = exponential

  var neuronConstantCached: Double = neuronConstant

  def receive = {

    case ParameterRequest(dataShardId) => {

      //log.info(s"Replica $dataShardId - Layer $layerId - Synact $inputId - $outputId read parameters!")
      context.sender() ! LatestParameter(coeff, exponential)
    }

    case ParameterRequestWithNeuronConstant(dataShardId) => {

      //log.info(s"Replica $dataShardId - Layer $layerId - Synact $inputId - $outputId read parameters!")
      context.sender() ! LatestParameterWithNeuronConstant(coeff, exponential, neuronConstant)
    }

    case CachedParameterRequest(dataShardId) => {

      //log.info(s"Replica $dataShardId - Layer $layerId - Synact $inputId - $outputId read parameters!")
      context.sender() ! LatestParameter(coeffCached, exponentialCached)
    }

    case CachedParameterRequestWithNeuronConstant(dataShardId) => {

      //log.info(s"Replica $dataShardId - Layer $layerId - Synact $inputId - $outputId read parameters!")
      context.sender() ! LatestParameterWithNeuronConstant(coeffCached, exponentialCached, neuronConstantCached)
    }

    case SynapseUpdateParameter(coeffChange, expChange) => {
      //1. Change Coeff
      var vectorChangeCorrected = coeffChange
      while (vectorChangeCorrected.length < coeff.length) vectorChangeCorrected = DenseVector(0.0d +: vectorChangeCorrected.toArray)
      coeff = coeff - vectorChangeCorrected * lambdaC

      //2. Change Exponential
      exponential = exponential - lambdaE * expChange
      if(exponential < 0) exponential = 0

      // Add term
      if(exponential > coeff.length) coeff = DenseVector(0.0d +: coeff.toArray)

    }

    case SynapseUpdateParameterWithNeuronConstant(coeffChange, expChange, neuronConstantChange) => {

      //0. Change Neuron Constant
      neuronConstant = neuronConstant - neuronConstantChange * lambdaC

      //1. Change Coeff
      var vectorChangeCorrected = coeffChange
      while (vectorChangeCorrected.length < coeff.length) vectorChangeCorrected = DenseVector(0.0d +: vectorChangeCorrected.toArray)
      coeff = coeff - vectorChangeCorrected * lambdaC

      //2. Change Exponential
      exponential = exponential - lambdaE * expChange
      if(exponential < 0) exponential = 0

      // Add term
      if(exponential > coeff.length) coeff = DenseVector(0.0d +: coeff.toArray)

    }

    case RefreshCachedParameters => {
      coeffCached = coeff
      exponentialCached = exponential
      neuronConstantCached = neuronConstant
      sender() ! ParametersCached(parameterShardId)
    }

    case OutputCoefficients => {
      val outputStringBuilder: StringBuilder = new StringBuilder
      coeff.toArray.reverse.zipWithIndex.map(x => if(x._2 == 0) outputStringBuilder ++= s"${x._1}x^$exponential" else outputStringBuilder ++= s" + ${x._1}x^${coeff.length - x._2}")
      log.info(s"Synapse of layer $layerId with input $inputId output $outputId: ${outputStringBuilder.toString()}")
      if(firstOfNeuron) log.info(s"Neuron Constant of layer $layerId neuron $outputId: $neuronConstant")
    }

    case OutputCachedCoefficients => {
      val outputStringBuilder: StringBuilder = new StringBuilder
      coeffCached.toArray.reverse.zipWithIndex.map(x => if(x._2 == 0) outputStringBuilder ++= s"${x._1}x^$exponentialCached" else outputStringBuilder ++= s" + ${x._1}x^${coeffCached.length - x._2}")
      log.info(s"Synapse of layer $layerId with input $inputId output $outputId: ${outputStringBuilder.toString()}")
      if(firstOfNeuron) log.info(s"Neuron Constant of layer $layerId neuron $outputId: $neuronConstantCached")
    }
  }
}
