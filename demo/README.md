# How to run the notebook

## Prerequisites

The following tools must be available on your system:

- **Babylon JDK** — built from this repo (see below)
- **LLVM 19** — `mlir-opt`, `mlir-translate`, `clang` at `/usr/lib/llvm-19/bin/`
- **Maven 3.x**
- **Python 3.x** with the `.venv` already created in the repo root

## 1 — Set JAVA_HOME to the Babylon JDK

The notebook requires the Babylon JDK built from this repository, not a stock JDK.

```bash
export JAVA_HOME=/home/bca/git/openjdk-babylon-mlir-tosa/build/linux-x86_64-server-release/images/jdk
```

Verify:

```bash
$JAVA_HOME/bin/java --version
# Expected: openjdk 26-internal-adhoc.bca.openjdk-babylon-mlir-tosa
```

## 2 — Build the JAR

From the repo root:

```bash
mvn package -f cr-examples/mlir-tosa/pom.xml -Pwith-examples -DskipTests
```

The output JAR is referenced by the notebook as:

```
../cr-examples/mlir-tosa/target/mlir-tosa-1.0-SNAPSHOT.jar
```

## 3 — Install Jupyter and the IJava kernel

Activate the venv created in the repo root:

```bash
source .venv/bin/activate
```

Install Jupyter:

```bash
pip install jupyter
```

Download and install the **IJava** kernel (a Jupyter kernel for Java/JShell):

```bash
# Download the latest IJava release
curl -L https://github.com/SpencerPark/IJava/releases/download/v1.3.0/ijava-1.3.0.zip -o /tmp/ijava.zip
unzip /tmp/ijava.zip -d /tmp/ijava

# Install into the active venv; JAVA_HOME tells install.py which java binary to use
python /tmp/ijava/install.py --sys-prefix
```

The kernel `kernel.json` is now at:

```
.venv/share/jupyter/kernels/java/kernel.json
```

Open it and add the two JVM flags (`--enable-preview` and `--add-modules jdk.incubator.code`)
immediately after the java binary in the `argv` array, so it looks like:

```json
{
  "argv": [
    "/path/to/babylon-jdk/bin/java",
    "--enable-preview",
    "--add-modules", "jdk.incubator.code",
    "-jar",
    "..."
  ],
  ...
}
```

Verify the kernel is registered:

```bash
jupyter kernelspec list
# Should show an entry named "java"
```

## 4 — Run the notebook

With the venv active and `JAVA_HOME` set:

```bash
jupyter notebook demo/demo.ipynb
```

Open the URL printed in the terminal. The notebook uses the **Java (Babylon)** kernel.

Run cells top to bottom. The first cell (`%jars ...`) loads the compiled JAR; the remaining cells
demonstrate the Babylon → TOSA → LLVM pipeline and the MNIST benchmark.

