from bench.bench import AnalogNASBench

ANALOG_BENCH_FILE = "analog_bench.csv"
SURROGATE_FILE = "surrogate.pth"

bench = AnalogNASBench(ANALOG_BENCH_FILE, SURROGATE_FILE, dataset="CIFAR-10")

print("Size of Trained dataset = ", bench.get_length())

config, arch = bench.query(1)

print(arch)

