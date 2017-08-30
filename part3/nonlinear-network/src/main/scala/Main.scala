/**
  * Created by LD on 2017-07-13.
  */
object Main {

  def main(args: Array[String]): Unit = {
    val LEARNING_RATE = 3.0
    val EPOCH = 30
    val MINI_BATCH_SIZE = 100
    val layer_structures = args.map(_.toInt)
    new NonlinearNN(LEARNING_RATE, layer_structures, EPOCH, MINI_BATCH_SIZE).start()
  }

}
