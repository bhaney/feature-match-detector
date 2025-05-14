import asyncio
import sys

from viam.module.module import Module
from viam.services.vision import Vision
from viam.resource.registry import Registry, ResourceCreatorRegistration
from src.featureMatchDetector import featureMatchDetector

async def main():
    """This function creates and starts a new module, after adding all desired resources.
    Resources must be pre-registered. For an example, see the `__init__.py` file.
    """
    Registry.register_resource_creator(
        Vision.API, 
        featureMatchDetector.MODEL,
        ResourceCreatorRegistration(
            featureMatchDetector.new, featureMatchDetector.validate
        ),
    )
    module = Module.from_args()
    module.add_model_from_registry(Vision.API, featureMatchDetector.MODEL)
    await module.start()

if __name__ == "__main__":
    asyncio.run(main())
