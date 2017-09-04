import Sequential._

import scala.io.StdIn

/**
  * Created by LD on 2017-02-06.
  */
object Main extends App{

  println("Select a simulation to run:\n" +
    "1.Singlr\n" +
    "2.Prdsn\n" +
    "3.Quad2\n" +
    "4.Multlr\n" +
    "5.Genpr\n" +
    "6.Gemn16\n" +
    "7.General Multilayer with AKKA, shard size 10\n" +
    "8.General Multilayer with AKKA, shard size 500\n" +
    "9.MNIST with AKKA, shard size 10000\n" +
    "")

  val simulation = StdIn.readInt()
  simulation match {
    case 1 => (new Singlr).MainRouting
    case 2 => (new Prdsn).MainRouting
    case 3 => (new Quad2).MainRouting
    case 4 => (new Multlr).MainRouting
    case 5 => (new Genpr).MainRouting
    case 6 => (new Gnmn16).MainRouting
    case 7 => new GeneralMultilayer(2, 1, -1, None, None, 10, false)
    case 8 => new GeneralMultilayer(2, 1, -1, None, None, 500, false)
    case 9 => new GeneralMultilayer(784, 10, -2, None, Some(5.0), 10000, true)
  }
}
