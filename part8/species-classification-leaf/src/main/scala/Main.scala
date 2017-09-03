/**
  * Created by LD on 2017-07-13.
  */
object Main {

  /**
    *
    * @param args For species classification by leaf data set, the input is size is 64, and output size is 100.
    *             e.g., a sample input for the main is '64 20 100', a three layers network
    */
  def main(args: Array[String]): Unit = {
    val LEARNING_RATE = 3.0
    val EPOCH = 30
    val MINI_BATCH_SIZE = 100
    val layer_structures = args.map(_.toInt)

    new LinearNN(LEARNING_RATE, layer_structures, EPOCH, MINI_BATCH_SIZE).start()
    new NonlinearNN(LEARNING_RATE, layer_structures, EPOCH, MINI_BATCH_SIZE).start()
    new SingleTermNN(LEARNING_RATE, layer_structures, EPOCH, MINI_BATCH_SIZE).start()
  }

}
