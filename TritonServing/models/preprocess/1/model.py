import triton_python_backend_utils as pb_utils
import json, os, cv2,base64, numpy as np, torch
from torch.utils.dlpack import from_dlpack, to_dlpack


class TritonPythonModel:
    """
    Your Python model must use the same class name. 
    Every Python model that is created must have "TritonPythonModel" as the class name.
    """

    def initalize(self, args):
        """
        `initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. 
        This function allowsthe model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
            Both keys and values are strings. The dictionary keys and values are:
            * model_config: A JSON string containing the model configuration
            * model_instance_kind: A string containing model instance kind
            * model_instance_device_id: A string containing model instance device ID
            * model repository : Model repository path
            * model_version: Model version
            * model_name: Model name
        """
        self.model_config = json.loads(args["model_config"])
        self.output_config = pb_utils.get_input_config_by_name(self.model_config,"output")
        self.output_dtype = pb_utils.get_datatype_from_triton(self.output_config["data_type"])
        # self.model_directory = os.path.dirname(os.path.realpath(__file__))
        

        print("Initalize...")

    def execute(self, requests):
        """
        `execute` must be implemented in every Python model. 
        `execute` function receives a list of pb_utils.InferenceRequest as the onlyargument.
        This function is called when an inference is requestedfor this model.
        Parameters
        ----------
        requests : list
            A list of pb_utils.InferenceRequest
        
        Returns
        -------
        list
            A list of pb_utils.InferenceResponse. The length of this list must be the same as `requests`
        """

        response = []
        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is

        #required.
        for request in requests:
            pass
            # Perform inference on the request and append it to responses list...
            input_tensor = pb_utils.get_input_tensor_by_name(request, "RAW_IMAGE").as_numpy()
            [height, width, _] = input_tensor.shape
            length = max((height, width))
            image = np.zeros((length, length, 3), np.uint8)
            image[0:height, 0:width] = input_tensor
            blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640))

            processed = pb_utils.Tensor("PREPROCESSED_OUTPUT",blob)
            inference_response = pb_utils.InferenceResponse(output_tensors=[processed])
            response.append(inference_response)

        # You must return a list of pb_utils. InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return response

    def finalize(self):
        """'finalizeis called only once when the model is being unloaded.
        Implementing `'finalize’ function is optional.
        This function allowsthe model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")