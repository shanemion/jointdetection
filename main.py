import sys
from air_writing_tracker import AirWritingTracker


def main():

    sentence = input("Sentence to write: ")

    tracker = AirWritingTracker(sentence)
    tracker.start()
    tracker.save_data("air_writing_data.npy")

if __name__ == "__main__":
    main()