package actors

import Utils.{InputOutput, SimulationType}
import actors.Checker.{CheckerDone, GatherOutput, OutputDone}
import actors.CheckerCoordinator.{CheckerCoordinatorStart, ParametersCached, RefreshCachedParameters}
import actors.DataShard.ReadyToProcess
import actors.Master.Exit
import actors.ParameterShard.OutputCachedCoefficients
import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import breeze.linalg.{DenseMatrix, DenseVector}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by LD on 2017-03-15.
  * The coordinator of all checkers
  */
object CheckerCoordinator {

  case class CheckerCoordinatorStart(startTime: Long)

  case object RefreshCachedParameters

  case class ParametersCached(parameterShardId: Int)
}


class CheckerCoordinator(dataShardActors:Seq[ActorRef],
                         dataShards: Seq[Seq[InputOutput]],
                         layers: Seq[Int],
                         parameterShards: Seq[Seq[Seq[ActorRef]]],
                         parameterShardCount: Int,
                         simType: SimulationType.Value) extends Actor with ActorLogging {

  val numberOfLayers = layers.size

  var parameterShardToUpdate = (0 until parameterShardCount).toSet

  var numberOfCheckerFinished = 0

  var allPass = true

  var passed = 0

  var totalSample = 0

  var startTime: Long = 0

  val checkerActors = dataShards.zipWithIndex.map { dataShard =>
    context.actorOf(Props(new Checker(
      checkId = dataShard._2,
      trainingData = dataShard._1,
      parameterShard = parameterShards,
      simType = simType
    )), s"Checker-${dataShard._2}")
  }

  def receive = {
    case CheckerCoordinatorStart(t) => {
      startTime = t
      numberOfCheckerFinished = 0
      parameterShards.foreach(_.foreach(_.foreach(_ ! RefreshCachedParameters)))
      context.become(waitForCacheDone)
    }
    case CheckerDone(checkerId, pass, positive) => {
      simType match {
        case SimulationType.Grid => {
          if(!pass) allPass = false
          numberOfCheckerFinished += 1
          //log.info(s"Checker Done: $numberOfCheckerFinished/${dataShards.size}")

          if(numberOfCheckerFinished == dataShards.size) {
            if(allPass) {
              //ALL DONE!
              parameterShards.foreach(_.foreach(_.foreach(_ ! OutputCachedCoefficients)))
              dataShardActors.foreach(context.stop)

              if (dataShards.head.head.input.length == 2) {
                val displayInput: ArrayBuffer[InputOutput] = ArrayBuffer[InputOutput]()
                val outputMatrix = DenseMatrix.zeros[Int](10,10)
                outputMatrix.foreachKey(x => displayInput += new InputOutput(DenseVector((x._1 + 1).toDouble / 10, (x._2 + 1).toDouble / 10), DenseVector(0.0)))
                checkerActors.head ! GatherOutput(displayInput.toArray.toSeq)
                context.become(waitOutputs)
              } else {
                log.info(s"${System.currentTimeMillis() - startTime}")
                //RunningTime.printStat()
                context.parent ! Exit
              }
              //context.system.terminate()
            }
            else {
              allPass = true
              self ! CheckerCoordinatorStart(startTime)
            }
          }
        }
        case SimulationType.Mnist => {
          passed += positive
          totalSample += dataShards(checkerId).size
          numberOfCheckerFinished += 1
          if(numberOfCheckerFinished == dataShards.size) {
            log.info(s"Passed: $passed $totalSample ${passed/totalSample}")
            log.info(s"${System.currentTimeMillis() - startTime}")
            context.parent ! Exit
          }
        }
      }
    }
  }

  def waitForCacheDone: Receive = {
    case ParametersCached(parameterShardId) => {
      parameterShardToUpdate -= parameterShardId
      if(parameterShardToUpdate.isEmpty){
        checkerActors.foreach(_ ! ReadyToProcess)
        context.unbecome()
      }
    }
  }

  def waitOutputs: Receive = {
    case OutputDone(checkerId, inputOutputs) => {
      val outputMatrix = DenseMatrix.zeros[Int](10,10)
      inputOutputs.foreach(x => outputMatrix((10*x.input(0)).toInt - 1,(10*x.input(1)).toInt - 1) = (x.output(0) + 0.5).toInt)
      log.info(s"\n$outputMatrix")
      log.info(s"${System.currentTimeMillis() - startTime}")
      //RunningTime.printStat()
      context.parent ! Exit
    }
  }
}
