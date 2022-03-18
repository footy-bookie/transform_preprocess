from xgb_processor import PreProcess


def main():
    print("Start with bookie_xgb_preprocess ms")
    pp = PreProcess()
    pp.data_for_predict()
    return print("Done with bookie_xgb_preprocess ms")


if __name__ == "__main__":
    main()
