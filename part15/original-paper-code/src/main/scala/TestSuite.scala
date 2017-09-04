import scala.io.StdIn

/**
  * Created by LD on 2017-03-21.
  */
object TestSuite extends App{


  println("Select a simulation to run:\n")

  val simulation = StdIn.readInt()

  simulation match {
    //Akka
    //Fixed Lambda = 5
    //2d - 7
    //2 layers
    case 1 => new GeneralMultilayer(4, 1, 5, Some(Seq(4, 2, 1)), Some(5), 10, false)

    //2d - 7
    //3 layers
    case 2 =>  new GeneralMultilayer(2, 1, 7, Some(Seq(2, 3, 1)), Some(5), 500, false)

    //2d - 7
    //4 layers
    case 3 => new GeneralMultilayer(2, 1, 7, Some(Seq(2, 4, 3, 1)), Some(5), 500, false)
  }

}
