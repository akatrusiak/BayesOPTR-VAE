Config merging is mostly done, but should be revisited later.
Look at core_functions/run_adapter.py to see how different "modes" are handled.

Run ouptuts are only going to one directory, this has not been thought of yet and might be tricky to have a handler that changes output dir, could also try CLI approach
Current style is to put everything possibly needed into the main "config.yaml" but in the future this should be broken into having specific sim_config.yaml and beam_config.yaml additions to the standard one. This was tricky so I have just stuck with one for now. 