package actors

import Utils.{Compute, InputOutput, SimulationType}
import actors.CheckerCoordinator.CheckerCoordinatorStart
import actors.DataShard.ReadyToProcess
import actors.Master.{Exit, ShardDone, Start}
import actors.ParameterShard.OutputCoefficients
import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import breeze.linalg.DenseVector

/**
  * Created by LD on 2017-02-26.
  */
object Master {

  case class ShardDone(shardId: Int)

  case object Start

  case object Exit

}

class Master(dataSet: Seq[InputOutput],
             validationSet: Option[Seq[InputOutput]],
             layers: Seq[Int],
             dataShardSize: Int,
             fixedLambda: Option[Double],
             lambdaOfC: Seq[Seq[Double]],
             lambdaOfE: Seq[Seq[Double]],
             simType: SimulationType.Value) extends Actor with ActorLogging{

  val numberOfLayers = layers.size

  //split data set into shards
  val dataShards = dataSet.grouped(dataShardSize).toSeq

  val validationShards = if(validationSet isDefined) validationSet.get.grouped(dataShardSize).toSeq else dataSet.grouped(dataShardSize).toSeq

  val parameterShardActors: Array[Array[Array[ActorRef]]] = new Array[Array[Array[ActorRef]]](numberOfLayers - 1)

  var parameterShardId: Int = -1

  for(i <- 1 until numberOfLayers) {
    val layerParameterShardActors = new Array[Array[ActorRef]](layers(i))
    for(j <- 0 until layers(i)) {
      val neuronParameterShardActors = new Array[ActorRef](layers(i - 1))
      for(k <- 0 until layers(i - 1)) {
        neuronParameterShardActors(k) = context.actorOf(Props(new ParameterShard(
          parameterShardId = {parameterShardId += 1; parameterShardId},
          layerId = i,
          inputId = k,
          outputId = j,
          lambdaC = if(fixedLambda.isDefined) fixedLambda.get else lambdaOfC(i)(j),
          lambdaE = if(fixedLambda.isDefined) fixedLambda.get else lambdaOfE(i)(j),
          initialCoeff = DenseVector(Array[Double](Compute.random2)),
          initialExpo = 1,
          firstOfNeuron = k == 0
        )), s"ParameterShard-$i-$k-$j")
      }
      layerParameterShardActors(j) = neuronParameterShardActors
    }
    parameterShardActors(i-1) = layerParameterShardActors
  }

  val dataShardActors = dataShards.zipWithIndex.map { dataShard =>
    context.actorOf(Props(new DataShard(
      shardId = dataShard._2,
      trainingData = dataShard._1,
      parameterShard = parameterShardActors.toSeq.map(_.toSeq.map(_.toSeq)),
      simType = simType
    )), s"DataShard-${dataShard._2}")
  }

  val checkerCoordinator = context.actorOf(Props(new CheckerCoordinator(
    dataShardActors = dataShardActors,
    dataShards = validationShards,
    layers = layers,
    parameterShards = parameterShardActors.toSeq.map(_.toSeq.map(_.toSeq)),
    parameterShardCount = parameterShardId,
    simType = simType
  )), s"CheckerCoordinator")

  var numberShardsFinished = 0

  var startTime: Long = _

  def receive = {

    case Start => {
      log.info("Start!")
      startTime = System.currentTimeMillis()
      dataShardActors.foreach(_ ! ReadyToProcess)
      if(simType == SimulationType.Grid) checkerCoordinator ! CheckerCoordinatorStart(startTime)
    }

    case ShardDone(shardId) => {
      numberShardsFinished += 1
      log.info(s"Finished: $numberShardsFinished/${dataShards.size}")

      if(numberShardsFinished == dataShards.size) {
        //ALL DONE!
        parameterShardActors.foreach(_.foreach(_.foreach(_ ! OutputCoefficients)))
        //context.system.terminate()
        if(simType == SimulationType.Mnist) checkerCoordinator ! CheckerCoordinatorStart(startTime)
      }
    }

    case Exit => {
      log.info(s"\nNumber of Data Shards: ${dataShards.size}")
      //context.system.terminate()
    }
  }
}
