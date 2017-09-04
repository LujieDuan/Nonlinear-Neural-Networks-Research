package actors.NLA

import Utils.Types._
import actors.NLA.Layer.{BackwardPass, ForwardPass}
import actors.NLA.Synact._
import akka.actor.{Actor, ActorRef}

/**
  * Created by LD on 2017-02-28.
  */
object Synact {

  case class SynactDoneFetch(layerId: Int, synactId: Int)

  case class NextSynact(next: ActorRef)

  case class UpdateParameter(coeff: Coeff)

  case class SynactForwardDone(synactId: Int, result: Double)

  case class SynactBackwardDone(synactId: Int)

  case class ParameterRequest(dataShardId: Int, layerId: Int, synactId: Int)

  case class LatestParameter(exp: Double, coeff: Coeff)

  case object FetchParameters

}


class Synact(replicaId: Int,
             layerId: Int,
             synactId: Int,
             parameterShard: ActorRef,
             layer: ActorRef,
             previousSynact: Option[ActorRef]) extends Actor {

  var latestCoeff: Coeff = _

  var latestExpo: Double = _

  var nextSynact: Option[ActorRef] = None

  def receive = {

    case NextSynact(next) => {
      nextSynact = Some(next)
    }

    case FetchParameters => {
      parameterShard ! ParameterRequest(replicaId, layerId, synactId)
      context.become(waitForParameters)
    }

    case ForwardPass(input, output) => {

      var result: Double = 0.0

      layer ! SynactForwardDone(synactId, result)
    }

    case BackwardPass(delats) => {


      layer ! SynactBackwardDone(synactId)
    }
  }

  def waitForParameters: Receive = {
    case LatestParameter(e, c) => {
      latestExpo = e
      latestCoeff = c
      layer ! SynactDoneFetch(layerId, synactId)
      context.unbecome()
    }
  }

}
