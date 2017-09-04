package actors.NLA

import Utils.Types._
import actors.DataShard.FetchParameters
import actors.NLA.Layer.{BackwardPass, ForwardPass, HigherLayer, LayerDoneFetch}
import actors.NLA.Synact.{NextSynact, SynactBackwardDone, SynactDoneFetch, SynactForwardDone}
import akka.actor.{Actor, ActorRef, Props}

/**
  * Created by LD on 2017-03-08.
  */
object Layer {

  case class ForwardPass(input: Seq[Double], output: Seq[Double])

  case class BackwardPass(deltas: Delta)

  case class LayerDoneFetch(layerId: Int)

  case class HigherLayer(higher: ActorRef)

}

class Layer(replicaId: Int,
            layerId: Int,
            lowerLayer: Option[ActorRef],
            parameterShard: Seq[ActorRef]) extends Actor {

  val numberOfSynacts = parameterShard.size

  val synacts: Array[ActorRef] = new Array[ActorRef](numberOfSynacts)

  for(s <- 0 until numberOfSynacts) {
    synacts(s) = context.actorOf(Props(new Synact(
      replicaId = replicaId,
      layerId = layerId,
      synactId = s,
      parameterShard = parameterShard(s),
      layer = self,
      previousSynact = if(s > 0) Some(synacts(s - 1)) else None
    )), s"Synact-$s")

    if(s > 0) synacts(s - 1) ! NextSynact(synacts(s))
  }

  var synactsToUpdate = (0 until numberOfSynacts).toSet

  var higherLayer: Option[ActorRef] = None


  def receive = {

    case HigherLayer(hl) => {
      higherLayer = Some(hl)
    }

    case FetchParameters => {
      synacts.foreach(_ ! FetchParameters)
      context.become(waitForFetchDone)
    }
  }

  def waitForFetchDone: Receive = {
    case SynactDoneFetch => {
      synactsToUpdate -= layerId
      if(synactsToUpdate.isEmpty) {
        context.parent ! LayerDoneFetch(layerId)
        //Wait the backward pass done so that can process the next input-output
        synactsToUpdate = (0 until numberOfSynacts).toSet
        context.become(readyToProcess)
      }
    }
  }

  def readyToProcess: Receive = {

    case ForwardPass(input, output) => {


      context.become(processing)
    }

    case BackwardPass(deltas) => {


      context.become(processing)
    }
  }

  def processing: Receive = {

    case SynactForwardDone(synactId, result) => {


      context.become(readyToProcess)
    }

    case SynactBackwardDone(i) => {


      context.unbecome()
    }

  }

}
