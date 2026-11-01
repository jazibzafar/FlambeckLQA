import argparse
from LQApy.modules import load_pretrained_maskrcnn_model
from LQApy.datasets import create_predict_loader
from LQApy.utils import save_preds_json
from LQApy.predict import predict


def get_pred_args():
    parser = argparse.ArgumentParser(description="Predicting configuration")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("oam_normalization", action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_pred_args()
    pred_model = load_pretrained_maskrcnn_model(args.model_path)
    dataloader = create_predict_loader(args.data_path, args.batch_size, args.num_workers, args.oam_normalization)
    predictions = predict(pred_model, dataloader, args.score_threshold)
    save_preds_json(predictions, args.save_path)
