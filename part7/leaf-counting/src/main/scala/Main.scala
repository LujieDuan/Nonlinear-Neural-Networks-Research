/**
  * Created by LD on 2017-07-13.
  */
object Main {

  /**
    *
    * @param args For species classification by leaf data set, the input is size is 49152, and output size is 1.
    *             e.g., a sample input for the main is '49152 20 1', a three layers network
    */
  def main(args: Array[String]): Unit = {
    val LEARNING_RATE = 3.0
    val EPOCH = 30
    val MINI_BATCH_SIZE = 100
    val layer_structures = args.map(_.toInt)

    new LinearNNRegression(LEARNING_RATE, layer_structures, EPOCH, MINI_BATCH_SIZE).start()
    new NonlinearNNRegression(LEARNING_RATE, layer_structures, EPOCH, MINI_BATCH_SIZE).start()
    new SingleTermNNRegression(LEARNING_RATE, layer_structures, EPOCH, MINI_BATCH_SIZE).start()
  }

}
