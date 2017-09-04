import Layer._
import Master.Exit
import Synapse._
import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import breeze.linalg.{*, DenseMatrix, DenseVector, sum}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Created by LD on 2017-05-24.
  */
object Layer {
  case class ForwardPass(input: DenseVector[Double], output: DenseVector[Double], update: Boolean, check: Boolean)

  case class BackwardPass(deltas: DenseVector[Double])

  case class HigherLayer(higher: ActorRef)

  case class FirstLayer(first: ActorRef)

  case class Process(training: Seq[(DenseVector[Double], DenseVector[Double])], test: Seq[(DenseVector[Double], DenseVector[Double])], epochs: Int, batchSize: Int)

  case class CheckNext(pass: Boolean)
}

class Layer(layerId: Int,
            lowerLayer: Option[ActorRef],
            eta: Double,
            sizes: Seq[Int]) extends Actor with ActorLogging{

  val numberOfNeurons = sizes(layerId + 1)

  val numberOfInputs = sizes(layerId)

  val numberOfSynapses = numberOfInputs * numberOfNeurons

  val synapses = new Array[Array[ActorRef]](numberOfNeurons)

  var lastLayerInput: DenseVector[Double] = _

  var finalTarget: DenseVector[Double] = _

  for(i <- 0 until numberOfNeurons) {
    synapses(i) = new Array[ActorRef](numberOfInputs)
    for(j <- 0 until numberOfInputs) {
      synapses(i)(j) = context.actorOf(Props(new Synapse(
        layerId = layerId,
        inputId = j,
        outputId = i,
        synapseId = i * numberOfInputs + j,
        eta = eta
      )), s"Synapse-$j-$i")
    }
  }

  var higherLayer: Option[ActorRef] = None

  var firstLayer: Option[ActorRef] = None

  //Matrix of output:
  var outputMatrix = DenseMatrix.zeros[Double](numberOfInputs, numberOfNeurons)

  var outputToUpdate = (0 until numberOfSynapses).toSet

  //Propagations vector
  var propagationMatrix = DenseMatrix.zeros[Double](numberOfNeurons, numberOfInputs)

  var propagationToUpdate = (0 until numberOfSynapses).toSet

  var trainingIndex = 0

  var trainingSet: Seq[(DenseVector[Double], DenseVector[Double])] = _

  var testIndex = 0

  var testSet: Seq[(DenseVector[Double], DenseVector[Double])] = _

  var iterationsDone = 0

  var totalIterations = 0

  var batchSize = 0

  var update = false

  var check = false

  var correct = 0

  def receive = {

    case HigherLayer(hl) => {
      higherLayer = Some(hl)
    }

    case FirstLayer(fl) => {
      firstLayer = Some(fl)
    }

    case Process(training, test, epochs, size) => {
      trainingSet = training
      testSet = test
      totalIterations = epochs
      iterationsDone = 0
      batchSize = size
      val next = trainingSet(trainingIndex)
      trainingIndex += 1
      assert(training.size > 1 && test.length > 1)
      val update = batchSize == 1
      ForwardPassFunc(next._1, next._2, update, false)
    }

    case ForwardPass(input, output, u, c) => {
      ForwardPassFunc(input, output, u, c)
    }

    case BackwardPass(deltas) => {
      BackwardPassFunc(deltas)
    }

    case CheckNext(pass) => {
      CheckNextFunc(pass)
    }

    case SynapseForwardDone(synapseId, inputId, outputId, result) => {

      outputMatrix(inputId, outputId) = result
      outputToUpdate -= synapseId
      if(outputToUpdate.isEmpty) {
        outputToUpdate = (0 until numberOfSynapses).toSet

        val layerOutputs = sum(outputMatrix(::, *)).t
        val activatedOutputs = Compute.activation(layerOutputs)

        higherLayer match {

          //If there is a higher layer, continue forward
          case Some(hl) => hl ! ForwardPass(activatedOutputs, finalTarget, update, check)

          case _ => {
            //compute output
            val finalOutput = activatedOutputs
            check match {
              case true => {
                //if this is the last layer, and this is a checker, check if result within error range
                var label =  finalTarget.toArray.indexOf(finalTarget.toArray.max)
                var predicate =  finalOutput.toArray.indexOf(finalOutput.toArray.max)
                val pass = label == predicate
                //if(!pass) log.info(s"Error: $label vs $predicate")
                label = 0
                predicate = 0
                firstLayer.get ! CheckNext(pass)
              }
              case false => {
                //if this is the last layer, and this is not a checker, turn around and do backward propagation
                //compute delta
                val dow: ArrayBuffer[Double] = new ArrayBuffer[Double]
                breeze.linalg.zipValues(finalTarget, finalOutput).foreach((target, output) => dow += (output - target) * output * (1 - output))
                //for synapses in this layer, do backward propagation first
                BackwardPassFunc(DenseVector(dow.toArray))
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
        propagationToUpdate = (0 until numberOfSynapses).toSet

        val propagationVector = sum(propagationMatrix(::, *)).t
        val activatedPropagation = Compute.activationDerivative(lastLayerInput) *:* propagationVector //elements-wise multiplication
        lowerLayer match {

          //If there is lower layer, continue
          case Some(ll) => ll ! BackwardPass(activatedPropagation)

          //All done, tell the data shard to learn the next input-output
          case _ =>
            if(trainingSet.size > trainingIndex){
              val next = trainingSet(trainingIndex)
              trainingIndex += 1
              val update = trainingIndex % batchSize == 0
              //if (update) log.info(s"${trainingIndex/batchSize}")
              ForwardPassFunc(next._1, next._2, update, false)
            }
            else
              CheckNextFunc(false)
        }
      }
    }
  }

  private def BackwardPassFunc(deltas: DenseVector[Double]) = {
    synapses.zipWithIndex.foreach(x => x._1.foreach(_ ! SynapseBackward(deltas(x._2), update)))
  }

  private def ForwardPassFunc(input: DenseVector[Double], output: DenseVector[Double], u: Boolean, c: Boolean) = {
    lastLayerInput = input
    finalTarget = output
    update = u
    check = c
    synapses.foreach(_.zipWithIndex.foreach(x => x._1 ! SynapseForward(input(x._2))))
  }

  private def CheckNextFunc(pass: Boolean) = {
    if (pass) correct += 1
    if (testSet.size > testIndex) {
      val next = testSet(testIndex)
      testIndex += 1
      ForwardPassFunc(next._1, next._2, false, true)
    } else {
      //synapses.foreach(_.foreach(x => x ! SynapseOutput))
      log.info(s"Passed: $correct/$testIndex.")
      correct = 0
      testIndex = 0
      if(totalIterations > iterationsDone){
        iterationsDone += 1
        trainingIndex = 0
        trainingSet = Random.shuffle(trainingSet)
        val next = trainingSet(trainingIndex)
        val update = batchSize == 1
        //if (update) log.info(s"${trainingIndex/batchSize}")
        trainingIndex += 1
        ForwardPassFunc(next._1, next._2, update, false)
      }
      else {
        context.parent ! Exit
      }
    }
  }
}
