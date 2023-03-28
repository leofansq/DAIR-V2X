import os.path as osp
import numpy as np
import torch.nn as nn
import logging
from sklearn.linear_model import LinearRegression
import pdb

logger = logging.getLogger(__name__)

from base_model import BaseModel
from model_utils import (
    init_model,
    inference_detector,
    inference_mono_3d_detector,
    BBoxList,
    EuclidianMatcher,
    SpaceCompensator,
    TimeCompensator,
    BasicFuser,
    CONSULT
)
from dataset.dataset_utils import (
    load_json,
    save_pkl,
    load_pkl,
    read_pcd,
    read_jpg,
)
from v2x_utils import (
    mkdir,
    get_arrow_end,
    box_translation,
    points_translation,
    get_trans,
    diff_label_filt,
)


def gen_pred_dict(id, timestamp, box, arrow, points, score, label, dist=None):
    if len(label) == 0:
        score = [-2333]
        label = [-1]
    save_dict = {
        "info": id,
        "timestamp": timestamp,
        "boxes_3d": box.tolist(),
        "arrows": arrow.tolist(),
        "scores_3d": score,
        "labels_3d": label,
        "points": points.tolist(),
        "dists_3d": dist
    }
    return save_dict


def get_box_info(result):
    if len(result[0]["boxes_3d"].tensor) == 0:
        box_lidar = np.zeros((1, 8, 3))
        box_ry = np.zeros(1)
    else:
        box_lidar = result[0]["boxes_3d"].corners.numpy()
        box_ry = result[0]["boxes_3d"].tensor[:, -1].numpy()
    box_centers_lidar = box_lidar.mean(axis=1)
    arrow_ends_lidar = get_arrow_end(box_centers_lidar, box_ry)
    return box_lidar, box_ry, box_centers_lidar, arrow_ends_lidar


class LateFusionInf(nn.Module):
    def __init__(self, args, pipe):
        super().__init__()
        self.model = None
        self.args = args
        self.pipe = pipe

    def pred(self, frame, trans, pred_filter):
        if self.args.sensortype == "lidar":
            id = frame.id["lidar"]
            logger.debug("infrastructure pointcloud_id: {}".format(id))
            path = osp.join(self.args.output, "inf", "lidar", id + ".pkl")
            frame_timestamp = frame["pointcloud_timestamp"]
        elif self.args.sensortype == "camera":
            id = frame.id["camera"]
            logger.debug("infrastructure image_id: {}".format(id))
            path = osp.join(self.args.output, "inf", "camera", id + ".pkl")
            frame_timestamp = frame["image_timestamp"]

        if osp.exists(path) and not self.args.overwrite_cache:
            pred_dict = load_pkl(path)
            return pred_dict, id

        logger.debug("prediction not found, predicting...")
        if self.model is None:
            raise Exception

        if self.args.sensortype == "lidar":
            tmp = frame.point_cloud(data_format="file")
            result, _ = inference_detector(self.model, tmp)
        elif self.args.sensortype == "camera":
            tmp = osp.join(self.args.input, "infrastructure-side", frame["image_path"])
            annos = osp.join(self.args.input, "infrastructure-side", "annos", id + ".json")
            result, _ = inference_mono_3d_detector(self.model, tmp, annos)
        box, box_ry, box_center, arrow_ends = get_box_info(result)

        ######
        dist = np.sqrt(np.sum(box_center[:,:2]**2,axis=1))

        # Convert to other coordinate
        if trans is not None:
            box = trans(box)
            box_center = trans(box_center)[:, np.newaxis, :]
            arrow_ends = trans(arrow_ends)[:, np.newaxis, :]

        # Filter out labels
        remain = []
        if len(result[0]["boxes_3d"].tensor) != 0:
            for i in range(box.shape[0]):
                if pred_filter(box[i]):
                    remain.append(i)

        # hard code by yuhb
        # TODO: new camera model
        if self.args.sensortype == "camera":
            for ii in range(len(result[0]["labels_3d"])):
                result[0]["labels_3d"][ii] = 2

        if len(remain) >= 1:
            box = box[remain]
            box_center = box_center[remain]
            arrow_ends = arrow_ends[remain]
            result[0]["scores_3d"] = result[0]["scores_3d"].numpy()[remain]
            result[0]["labels_3d"] = result[0]["labels_3d"].numpy()[remain]
            dist = dist[remain]
        else:
            box = np.zeros((1, 8, 3))
            box_center = np.zeros((1, 1, 3))
            arrow_ends = np.zeros((1, 1, 3))
            result[0]["labels_3d"] = np.zeros((1))
            result[0]["scores_3d"] = np.zeros((1))
            dist = np.zeros((1))

        if self.args.sensortype == "lidar" and self.args.save_point_cloud:
            save_data = trans(frame.point_cloud(format="array"))
        elif self.args.sensortype == "camera" and self.args.save_image:
            save_data = frame.image(data_format="array")
        else:
            save_data = np.array([])

        pred_dict = gen_pred_dict(
            id,
            frame_timestamp,
            box,
            np.concatenate([box_center, arrow_ends], axis=1),
            save_data,
            result[0]["scores_3d"].tolist(),
            result[0]["labels_3d"].tolist(),
            dist
        )
        save_pkl(pred_dict, path)

        return pred_dict, id

    def forward(self, data, trans, pred_filter, prev_inf_frame_func=None, is_count_byte=True):
        try:
            pred_dict, id = self.pred(data, trans, pred_filter)
        except Exception:
            logger.info("building model")
            self.model = init_model(
                self.args.inf_config_path,
                self.args.inf_model_path,
                device=self.args.device,
            )
            pred_dict, id = self.pred(data, trans, pred_filter)
        self.pipe.send("boxes", pred_dict["boxes_3d"], is_count_byte)
        self.pipe.send("score", pred_dict["scores_3d"], is_count_byte)
        self.pipe.send("label", pred_dict["labels_3d"], is_count_byte)
        self.pipe.send("dist", pred_dict["dists_3d"], is_count_byte)


        if prev_inf_frame_func is not None:
            prev_frame, delta_t = prev_inf_frame_func(id, sensortype=self.args.sensortype)
            if prev_frame is not None:
                prev_frame_trans = prev_frame.transform(from_coord="Infrastructure_lidar", to_coord="Vehicle_lidar")
                prev_frame_trans.veh_name = trans.veh_name
                prev_frame_trans.delta_x = trans.delta_x
                prev_frame_trans.delta_y = trans.delta_y
                try:
                    pred_dict, _ = self.pred(
                        prev_frame,
                        prev_frame_trans,
                        pred_filter,
                    )
                except Exception:
                    logger.info("building model")
                    self.model = init_model(
                        self.args.inf_config_path,
                        self.args.inf_model_path,
                        device=self.args.device,
                    )
                    pred_dict, _ = self.pred(
                        prev_frame,
                        prev_frame_trans,
                        pred_filter,
                    )
                self.pipe.send("prev_boxes", pred_dict["boxes_3d"], is_count_byte)
                self.pipe.send("prev_time_diff", delta_t)
                self.pipe.send("prev_label", pred_dict["labels_3d"], is_count_byte)
                self.pipe.send("prev_dist", pred_dict["dists_3d"], is_count_byte)


        return id


class LateFusionVeh(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = None
        self.args = args

    def pred(self, frame, trans, pred_filter):
        if self.args.sensortype == "lidar":
            id = frame.id["lidar"]
            logger.debug("vehicle pointcloud_id: {}".format(id))
            path = osp.join(self.args.output, "veh", "lidar", id + ".pkl")
            frame_timestamp = frame["pointcloud_timestamp"]
        elif self.args.sensortype == "camera":
            id = frame.id["camera"]
            logger.debug("vehicle image_id: {}".format(id))
            path = osp.join(self.args.output, "veh", "camera", id + ".pkl")
            frame_timestamp = frame["image_timestamp"]

        if osp.exists(path) and not self.args.overwrite_cache:
            pred_dict = load_pkl(path)
            return pred_dict, id

        logger.debug("prediction not found, predicting...")
        if self.model is None:
            raise Exception

        if self.args.sensortype == "lidar":
            tmp = frame.point_cloud(data_format="file")
            result, _ = inference_detector(self.model, tmp)
        elif self.args.sensortype == "camera":
            tmp = osp.join(self.args.input, "vehicle-side", frame["image_path"])
            annos = osp.join(self.args.input, "vehicle-side", "annos", id + ".json")
            result, _ = inference_mono_3d_detector(self.model, tmp, annos)
        box, box_ry, box_center, arrow_ends = get_box_info(result)

        ######
        dist = np.sqrt(np.sum(box_center[:, :2]**2,axis=1))

        # Convert to other coordinate
        if trans is not None:
            box = trans(box)
            box_center = trans(box_center)[:, np.newaxis, :]
            arrow_ends = trans(arrow_ends)[:, np.newaxis, :]

        # Filter out labels
        remain = []
        if len(result[0]["boxes_3d"].tensor) != 0:
            for i in range(box.shape[0]):
                if pred_filter(box[i]):
                    remain.append(i)

        # hard code by yuhb
        # TODO: new camera model
        if self.args.sensortype == "camera":
            for ii in range(len(result[0]["labels_3d"])):
                result[0]["labels_3d"][ii] = 2

        if len(remain) >= 1:
            box = box[remain]
            box_center = box_center[remain]
            arrow_ends = arrow_ends[remain]
            result[0]["scores_3d"] = result[0]["scores_3d"].numpy()[remain]
            result[0]["labels_3d"] = result[0]["labels_3d"].numpy()[remain]
            dist = dist[remain]
        else:
            box = np.zeros((1, 8, 3))
            box_center = np.zeros((1, 1, 3))
            arrow_ends = np.zeros((1, 1, 3))
            result[0]["labels_3d"] = np.zeros((1))
            result[0]["scores_3d"] = np.zeros((1))
            dist = np.zeros((1))

        if self.args.sensortype == "lidar" and self.args.save_point_cloud:
            save_data = trans(frame.point_cloud(format="array"))
        elif self.args.sensortype == "camera" and self.args.save_image:
            save_data = frame.image(data_format="array")
        else:
            save_data = np.array([])

        pred_dict = gen_pred_dict(
            id,
            frame_timestamp,
            box,
            np.concatenate([box_center, arrow_ends], axis=1),
            save_data,
            result[0]["scores_3d"].tolist(),
            result[0]["labels_3d"].tolist(),
            dist
        )
        save_pkl(pred_dict, path)

        return pred_dict, id

    def forward(self, data, trans, pred_filter):
        try:
            pred_dict, id = self.pred(data, trans, pred_filter)
        except Exception:
            logger.info("building model")
            self.model = init_model(
                self.args.veh_config_path,
                self.args.veh_model_path,
                device=self.args.device,
            )
            pred_dict, id = self.pred(data, trans, pred_filter)
        return pred_dict, id


class LateFusion(BaseModel):
    def add_arguments(parser):
        parser.add_argument("--inf-config-path", type=str, default="")
        parser.add_argument("--inf-model-path", type=str, default="")
        parser.add_argument("--veh-config-path", type=str, default="")
        parser.add_argument("--veh-model-path", type=str, default="")
        parser.add_argument("--no-comp", action="store_true")
        parser.add_argument("--overwrite-cache", action="store_true")

    def __init__(self, args, pipe):
        super().__init__()
        self.pipe = pipe
        self.inf_model = LateFusionInf(args, pipe)
        self.veh_model = LateFusionVeh(args)
        self.args = args
        self.space_compensator = SpaceCompensator()
        self.time_compensator = TimeCompensator(EuclidianMatcher(diff_label_filt))
        mkdir(args.output)
        mkdir(osp.join(args.output, "inf"))
        mkdir(osp.join(args.output, "veh"))
        mkdir(osp.join(args.output, "inf", "lidar"))
        mkdir(osp.join(args.output, "veh", "lidar"))
        mkdir(osp.join(args.output, "inf", "camera"))
        mkdir(osp.join(args.output, "veh", "camera"))
        mkdir(osp.join(args.output, "result"))

        self.perspective = "vehicle"

    def forward(self, vic_frame, filt, prev_inf_frame_func=None, prev_vic_frame_func=None, *args):

        pred_inf, pred_veh, id_inf, id_veh = self.pred(vic_frame, filt, prev_inf_frame_func)

        # matcher = EuclidianMatcher(diff_label_filt)
        # ind_inf, ind_veh, cost = matcher.match(pred_inf, pred_veh)
        # logger.debug("matched boxes: {}, {}".format(ind_inf, ind_veh))
        # fuser = BasicFuser(perspective="vehicle", trust_type="main", retain_type="all")
        # result = fuser.fuse(pred_inf, pred_veh, ind_inf, ind_veh, self.model_confidence)

        # CONSULT
        matcher = EuclidianMatcher(diff_label_filt)
        fuser = CONSULT(self.perspective)

        if self.args.sensortype == 'lidar':
            self.model_confidence = min(17.58 / 48.06, 1.0)
            tf_para = {"type":"max", "prev_th":0.5, "prev_decay":0.5}
        elif self.args.sensortype == 'camera':
            self.model_confidence = min(14.02/9.03, 1.0)
            tf_para = {"type":"main", "prev_th":0.8, "prev_decay":0.6}
        
        pred_temporal = self.temporal_predict(prev_vic_frame_func, id_veh, filt)
        if pred_temporal is not None:
            print ("Found previous frames, CONSULT-temporal is active.")
            if self.perspective=="vehicle":
                ind_temporal, ind_cur, _ = matcher.match(pred_temporal, pred_veh)
                result = fuser.fuse(pred_veh, pred_temporal, ind_cur, ind_temporal, fusion_type='temporal',\
                                    temporal_type=tf_para["type"], temporal_th=tf_para["prev_th"], temporal_decay=tf_para["prev_decay"])
                pred_veh = BBoxList(
                                np.array(result["boxes_3d"]),
                                None,
                                np.array(result["labels_3d"]),
                                np.array(result["scores_3d"]),
                                np.array(result["dists_3d"]),
                            )
            elif self.perspective=="infrastructure":
                ind_temporal, ind_cur, _ = matcher.match(pred_temporal, pred_inf)
                result = fuser.fuse(pred_inf, pred_temporal, ind_inf, ind_temporal, fusion_type='temporal',\
                                    temporal_type=tf_para["type"], temporal_th=tf_para["prev_th"], temporal_decay=tf_para["prev_decay"])
                pred_inf = BBoxList(
                                np.array(result["boxes_3d"]),
                                None,
                                np.array(result["labels_3d"]),
                                np.array(result["scores_3d"]),
                                np.array(result["dists_3d"]),
                            )     

        ind_inf, ind_veh, _ = matcher.match(pred_inf, pred_veh)
        logger.debug("matched boxes: {}, {}".format(ind_inf, ind_veh))
        
        result = fuser.fuse(pred_inf, pred_veh, ind_inf, ind_veh, self.model_confidence, 'spatial')


        result["inf_id"] = id_inf
        result["veh_id"] = id_veh
        result["inf_boxes"] = pred_inf.boxes
        return result
    
    def pred(self, vic_frame, filt, prev_inf_frame_func, is_count_byte=True):
        id_inf = self.inf_model(
            vic_frame.infrastructure_frame(),
            vic_frame.transform(from_coord="Infrastructure_lidar", to_coord="Vehicle_lidar"),
            filt,
            prev_inf_frame_func if not self.args.no_comp else None,
            is_count_byte
        )
        pred_dict, id_veh = self.veh_model(vic_frame.vehicle_frame(), None, filt)

        # logger.info("running late fusion...")
        pred_inf = BBoxList(
            np.array(self.pipe.receive("boxes")),
            None,
            np.array(self.pipe.receive("label")),
            np.array(self.pipe.receive("score")),
            np.array(self.pipe.receive("dist")),
        )
        pred_veh = BBoxList(
            np.array(pred_dict["boxes_3d"]),
            None,
            np.array(pred_dict["labels_3d"]),
            np.array(pred_dict["scores_3d"]),
            np.array(pred_dict["dists_3d"]),
        )
        if (vic_frame.time_diff > 0) and (prev_inf_frame_func is not None) and (not self.args.no_comp):
            if self.pipe.receive("prev_boxes") is not None:
                pred_inf_prev = BBoxList(
                    np.array(self.pipe.receive("prev_boxes")),
                    None,
                    np.array(self.pipe.receive("prev_label")),
                    None,
                    np.array(self.pipe.receive("prev_dist")),
                )
                offset = self.time_compensator.compensate(
                    pred_inf_prev,
                    pred_inf,
                    self.pipe.receive("prev_time_diff"),
                    vic_frame.time_diff,
                )
                pred_inf.move_center(offset)
                logger.debug("time compensation: {}".format(offset))
            else:
                print("no previous frame found, time compensation is skipped")
        
        return pred_inf, pred_veh, id_inf, id_veh
    
    def temporal_predict(self, prev_vic_frame_func, cur_id, filt):

        p_vic = prev_vic_frame_func(cur_id, sensortype=self.args.sensortype, specified_k=1)

        p_frame, p_delta = p_vic["frame_vic"], p_vic["delta_t"]

        temporal_pred = None

        if p_frame is not None:
            p_id = p_frame.veh_frame.id[self.args.sensortype]

            pp_vic = prev_vic_frame_func(p_id, sensortype=self.args.sensortype, specified_k=1)
            pp_frame, pp_delta = pp_vic["frame_vic"], pp_vic["delta_t"]

            if pp_frame is not None:


                matcher = EuclidianMatcher(diff_label_filt)
                fuser = CONSULT(perspective=self.perspective)

                preds = []
                for frame_i in [pp_frame, p_frame]:
                    
                    pred_inf, pred_veh, _, _ = self.pred(frame_i, filt, None, False)
                    ind_inf, ind_veh, _ = matcher.match(pred_inf, pred_veh)                
                    pred = fuser.fuse(pred_inf, pred_veh, ind_inf, ind_veh, self.model_confidence, 'spatial')
                    pred = BBoxList(
                                    pred["boxes_3d"],
                                    None,
                                    pred["labels_3d"],
                                    pred["scores_3d"],
                                    np.ones_like(pred["labels_3d"])*1e6,
                                )
                    preds.append(pred)
                
                ind_pp, ind_p, _ = matcher.match(preds[0], preds[1])
                # general motion prediction
                avg_offset = (np.mean(preds[1].center, axis=0) - np.mean(preds[0].center, axis=0)) * p_delta/pp_delta
                offset = np.ones((preds[1].num_boxes, 2))
                offset[:, 0] *= avg_offset[0]
                offset[:, 1] *= avg_offset[1]
                # matched object interpolation
                offset[ind_p] = (preds[1].center[ind_p][:, :2] - preds[0].center[ind_pp][:, :2]) * p_delta/pp_delta
                # matched object confidence adjustment
                preds[1].confidence[ind_p] = np.min([preds[1].confidence[ind_p], preds[0].confidence[ind_pp]], axis=0)

                preds[1].move_center(offset)
                temporal_pred = preds[1]     
        return temporal_pred

