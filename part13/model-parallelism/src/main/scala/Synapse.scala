import Synapse._
import akka.actor.{Actor, ActorLogging}
import breeze.linalg.sum
import breeze.stats.distributions.Rand

/**
  * Created by LD on 2017-05-24.
  */
object Synapse {
  case class SynapseForward(input: Double)

  case class SynapseForwardDone(synapseId: Int, inputId: Int, outputId: Int, output: Double)

  case class SynapseBackward(propagation: Double, update: Boolean)

  case class SynapseBackwardDone(synapseId: Int, inputId: Int, outputId: Int, sum: Double)

  case object SynapseOutput
}

class Synapse(layerId: Int,
              inputId: Int,
              outputId: Int,
              synapseId: Int,
              eta: Double) extends Actor with ActorLogging{

  var latestCoeff: Seq[Double] = Seq(Rand.gaussian.draw())

  var latestExpo: Double = 1.0d

  var latestNeuronConstant: Double = Rand.gaussian.draw()

  var firstInputOfNeuron: Boolean = inputId == 0

  var latestInputs: Double = _

  var batchCoeffChange: Seq[Double] = Seq.fill(latestCoeff.length)(0.0d)

  var batchExpChange: Double = 0.0

  var batchConstantChange: Double = 0.0

  var batchSize: Int =  0

  def receive = {

    case SynapseForward(input) => {
      latestInputs = input
      val polynomialTerms: Seq[Double] = Compute.pow(input, latestExpo) * latestCoeff(0) +: (latestCoeff.length - 1 until 0 by -1).map(x => Compute.pow(input, x) * latestCoeff(x))
      val polynomialResult: Double = sum(polynomialTerms)
      val result: Double = firstInputOfNeuron match {
        case true => polynomialResult + latestNeuronConstant
        case false => polynomialResult
      }
      context.parent ! SynapseForwardDone(synapseId, inputId, outputId, result)
    }

    case SynapseBackward(deltas, update) => {
      //First, compute the deltas/changes
      val coeffChange = (deltas * Compute.pow(latestInputs, latestExpo)) +: (latestCoeff.length - 1 until 0 by -1).map(x => deltas * Compute.pow(latestInputs, x))
      val expChange = deltas * latestCoeff(0) * Compute.pow(latestInputs, latestExpo) * Compute.lan(latestInputs)
      val neuronConstantChange: Double = deltas
      batchCoeffChange = (batchCoeffChange, coeffChange).zipped.map((x, y) => x + y)
      batchExpChange += expChange
      batchConstantChange += neuronConstantChange
      batchSize += 1


      if(update) {
        //0. Change Neuron Constant
        if (firstInputOfNeuron) latestNeuronConstant = latestNeuronConstant - (batchConstantChange / batchSize) * eta

        //1. Change Coeff
        latestCoeff = (latestCoeff, batchCoeffChange).zipped.map((oldValue, newValue) => oldValue - (newValue / batchSize) * eta)

        //2. Change Exponential
        latestExpo = latestExpo - eta * (batchExpChange / batchSize)
        if(latestExpo < 0) latestExpo = 0

        // Add term
        if(latestExpo > latestCoeff.length) latestCoeff = 0.0d +: latestCoeff

        batchCoeffChange = Seq.fill(latestCoeff.length)(0.0d)
        batchExpChange = 0.0
        batchConstantChange = 0.0
        batchSize = 0
      }


      //Second, compute the sum/dow values for lower layer to propagate
      val expVector = (latestCoeff(0) * latestExpo * Compute.pow(latestInputs, latestExpo - 1)) +: (latestCoeff.length - 1 until 0 by -1).map(x => x * Compute.pow(latestInputs, x - 1) * latestCoeff(x))
      val prevsum: Double = deltas * sum(expVector)

      context.parent ! SynapseBackwardDone(synapseId, inputId, outputId, prevsum)
    }

    case SynapseOutput => {
      val outputStringBuilder: StringBuilder = new StringBuilder
      latestCoeff.toArray.reverse.zipWithIndex.map(x => if(x._2 == 0) outputStringBuilder ++= s"${x._1}x^$latestExpo" else outputStringBuilder ++= s" + ${x._1}x^${latestCoeff.length - x._2}")
      log.info(s"Synapse of layer $layerId with input $inputId output $outputId: ${outputStringBuilder.toString()}")
      if(firstInputOfNeuron) log.info(s"Neuron Constant of layer $layerId neuron $outputId: $latestNeuronConstant")
    }
  }
}
