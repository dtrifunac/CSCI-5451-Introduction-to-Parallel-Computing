#!/bin/bash

if [[ -n "$DEBUG" ]]; then
    DEBUGOPTS="-x DEBUG"
fi

export MPIOPTS=""
MPIOPTS+="$DEBUGOPTS"
MPIOPTS+=" --oversubscribe"
MPIOPTS+=" --mca opal_warn_on_missing_libcuda 0"
MPIOPTS+=" --mca mca_base_component_show_load_errors 0"
# export MPIOPTS="$DEBUGOPTS --mca opal_warn_on_missing_libcuda 0"
