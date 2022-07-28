from utility import History
import random
import time

def test1():
    """Test progress bar"""
    epochs = 3
    size = 100
    history = History(size, epochs, progress_bar = False)
    for i in range(epochs):
        history.new_epoch()
        for j in range(size):
            history.increment(j, random.random(), random.random())
            time.sleep(.000001)
        history.validate(random.random(), random.random())

    # Local machine: python main.py --progress_bar=TRUE
    # TACC: python ${TACC_WORKDIR}/main.py --user_dir=${TACC_USERDIR}/

def test2():
    pass


if __name__ == "__main__":
    test2()