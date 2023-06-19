import triton_python_backend_utils as pb_utils
import numpy as np, cv2

class TritonPythonModel:

    def execute(self, requests):

        response = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "POSTPROCESS_INPUT").as_numpy()
            pred = cv_nms(input_tensor)
            output_tensor = pb_utils.Tensor("POSTPROCESSED_OUTPUT", pred.astype(np.float32))
            inference_response = pb_utils.InferenceResponse([output_tensor])
            response.append(inference_response)

        return response

def cv_nms(outputs, conf_thres=0.25, iou_thres=0.45):
    boxes = []
    scores = []
    class_ids = []
    rows = outputs.shape[1]

    for i in range(rows):
        classes_scores = outputs[0][i][5:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        maxScore *= outputs[0][i][4]
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]), 
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2], outputs[0][i][3]]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    indexes = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres, 0.5)
    result_boxes = np.array(boxes)[indexes]
    result_scores = np.array(scores).reshape(-1,1)[indexes]
    result_class_ids = np.array(class_ids).reshape(-1,1)[indexes]
    return np.concatenate((result_boxes,result_scores,result_class_ids), axis=1)