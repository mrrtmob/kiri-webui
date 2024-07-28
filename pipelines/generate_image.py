import os
import boto3
import requests
from typing import Literal, Optional
from openai import OpenAI
from blueprints.function_calling_blueprint import Pipeline as FunctionCallingBlueprint
from datetime import datetime


class Pipeline(FunctionCallingBlueprint):
    class Valves(FunctionCallingBlueprint.Valves):
        OPENAI_API_BASE_URL: str = "https://api.openai.com/v1"
        OPENAI_API_KEY: str = ""
        IMAGE_SIZE: Literal[
            "256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"
        ] = "512x512"
        NUM_IMAGES: int = 1
        IMAGE_MODEL: Literal["dall-e-2", "dall-e-3"] = "dall-e-2"
        AWS_ACCESS_KEY_ID: str = ""
        AWS_SECRET_ACCESS_KEY: str = ""
        S3_BUCKET_NAME: str = ""

    class Tools:
        def __init__(self, pipeline) -> None:
            self.pipeline = pipeline

        def ask_details_for_image_generation(
            self, prompt: str, is_details: bool
        ) -> str:
            """
            Generates an image based on a text description. Provide a detailed text description of the image you want to create.

            :param is_details: return True if user ask to generate image else False
            :param prompt: A text description of the desired image.
            :return: A string of generated image URLs or a question to as for a details.
            """

            if not is_details:
                return "To create the best possible image, it would be helpful if you could provide some details about what you'd like to see in the image"

            if not self.pipeline.valves.OPENAI_API_KEY:
                return "OpenAI API Key not set. Please set it up in the valves."

            # Check if prompt is empty
            if not prompt.strip():
                return ""

            try:
                client = OpenAI(
                    api_key=self.pipeline.valves.OPENAI_API_KEY,
                    base_url=self.pipeline.valves.OPENAI_API_BASE_URL,
                )

                # Generate the image
                response = client.images.generate(
                    model=self.pipeline.valves.IMAGE_MODEL,
                    prompt=prompt,
                    size=self.pipeline.valves.IMAGE_SIZE,
                    n=self.pipeline.valves.NUM_IMAGES,
                )

                print(response)  # Debugging line to see the full response

                if (
                    response.data
                    and len(response.data) > 0
                    and hasattr(response.data[0], "url")
                ):
                    openai_image_url = response.data[0].url

                    # Save image to S3
                    try:
                        # Download the image
                        image_response = requests.get(openai_image_url)
                        image_response.raise_for_status()

                        # Create an S3 client
                        s3_client = boto3.client(
                            "s3",
                            aws_access_key_id=self.pipeline.valves.AWS_ACCESS_KEY_ID,
                            aws_secret_access_key=self.pipeline.valves.AWS_SECRET_ACCESS_KEY,
                        )

                        # Generate a unique filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{prompt[:10].replace(' ', '_')}_{timestamp}_{os.urandom(8).hex()}.png"

                        # Upload the image to S3
                        s3_client.put_object(
                            Bucket=self.pipeline.valves.S3_BUCKET_NAME,
                            Key=filename,
                            Body=image_response.content,
                            ContentType="image/png",
                        )

                        # Generate the S3 URL
                        s3_url = f"https://{self.pipeline.valves.S3_BUCKET_NAME}.s3.amazonaws.com/{filename}"

                        return f"Image URL generated: {s3_url}"
                    except Exception as e:
                        return f"Error saving image to S3: {str(e)}"
                else:
                    return "No image URL was generated or response structure is unexpected."

            except Exception as e:
                return f"Error generating image: {str(e)}"

    def __init__(self):
        super().__init__()

        self.name = "generate_image"
        self.valves = self.Valves(
            **{
                **self.valves.model_dump(),
                "pipelines": ["*"],  # Connect to all pipelines
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
                "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", ""),
                "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
                "S3_BUCKET_NAME": os.getenv("S3_BUCKET_NAME", ""),
            },
        )
        self.tools = self.Tools(self)
