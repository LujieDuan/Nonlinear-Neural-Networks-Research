package Utils

/**
  * Created by LD on 2017-02-10.
  *
  * The following code borrow from:
  * http://alvinalexander.com/scala/how-to-open-read-text-files-in-scala-cookbook-examples
  *
  */
object Control {

  def using[A <: { def close(): Unit }, B](resource: A)(f: A => B): B =
    try {
      f(resource)
    } finally {
      resource.close()
    }
}
