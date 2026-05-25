#!/bin/bash
# List PipeWire/PulseAudio audio output sinks
pw-cli list-objects 2>/dev/null | awk '
/node.description =/ { desc=$0; sub(/.*= "/,"",desc); sub(/"$/,"",desc) }
/node.name =/ { name=$0; sub(/.*= "/,"",name); sub(/"$/,"",name) }
/node.nick =/ { nick=$0; sub(/.*= "/,"",nick); sub(/"$/,"",nick) }
/media.class = "Audio\/Sink"/ {
    if (name != "") {
        printf "  %-35s\n    --audio-device %s\n\n", nick " (" desc ")", name
    }
    desc=""; name=""; nick=""
}
'
