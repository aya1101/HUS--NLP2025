ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.12"
var ThisBuild / sparkVersion = "4.0.0" 

lazy val root = (project in file("."))
  .settings(
    name := "spark-nlp-labs",
    // Enable forking a new JVM for 'run' and 'test' tasks
    // This is necessary to pass JVM options for Spark on Java 9+
    fork := true,
    // Add JVM options to allow Spark to access internal Java APIs
    javaOptions ++= Seq(
      "--add-opens=java.base/java.nio=ALL-UNNAMED",
      "--add-opens=java.base/java.nio.channels=ALL-UNNAMED",
      "--add-opens=java.base/java.lang=ALL-UNNAMED",
      "--add-opens=java.base/java.io=ALL-UNNAMED",
      "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"
    ),
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % sparkVersion,
      "org.apache.spark" %% "spark-sql" % sparkVersion,
      "org.apache.spark" %% "spark-mllib" % sparkVersion
    )
  )
