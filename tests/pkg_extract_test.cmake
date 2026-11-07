# Makes a fake PKG, extracts it with lpn-pkg and compares the result with what went in.
file(REMOVE_RECURSE "${WORK}")
file(MAKE_DIRECTORY "${WORK}")
execute_process(COMMAND "${PYTHON}" -I "${SRC}/tests/make_fake_pkg.py" "${SRC}/src/pkg/keys.h" "${WORK}/test.pkg"
                        "${WORK}/expected" RESULT_VARIABLE r)
if(NOT r EQUAL 0)
    message(FATAL_ERROR "make_fake_pkg.py failed")
endif()
execute_process(COMMAND "${LPN_PKG}" extract "${WORK}/test.pkg" "${WORK}/game" RESULT_VARIABLE r)
if(NOT r EQUAL 0)
    message(FATAL_ERROR "lpn-pkg extract failed")
endif()
execute_process(COMMAND diff -r "${WORK}/expected" "${WORK}/game/CUSA12776" RESULT_VARIABLE r)
if(NOT r EQUAL 0)
    message(FATAL_ERROR "extracted files differ")
endif()
# A second extraction must not touch the folder the first made.
execute_process(COMMAND "${LPN_PKG}" extract "${WORK}/test.pkg" "${WORK}/game" RESULT_VARIABLE r ERROR_QUIET OUTPUT_QUIET)
if(r EQUAL 0)
    message(FATAL_ERROR "lpn-pkg extracted over an existing folder")
endif()

# An update and DLC, from .pkg files and from an unpacked folder (pkg/install.h).
macro(run)
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE r OUTPUT_QUIET)
    if(NOT r EQUAL 0)
        message(FATAL_ERROR "failed: ${ARGN}")
    endif()
endmacro()
foreach(kind update dlc)
    run("${PYTHON}" -I "${SRC}/tests/make_fake_pkg.py" "${SRC}/src/pkg/keys.h" "${WORK}/${kind}.pkg"
        "${WORK}/expected-${kind}" ${kind})
endforeach()
run("${LPN_PKG}" install "${WORK}/update.pkg" "${WORK}/game")
run("${LPN_PKG}" install "${WORK}/dlc.pkg" "${WORK}/game")
# The updated version: the game with the update's files over it; the game itself unchanged.
file(COPY "${WORK}/expected/" DESTINATION "${WORK}/expected-updated")
# file(COPY) skips a file whose destination has the same mtime to the second, which the game's and
# the update's files often do: the update's files are removed first so they are always copied.
file(GLOB_RECURSE patched RELATIVE "${WORK}/expected-update" "${WORK}/expected-update/*")
foreach(f ${patched})
    file(REMOVE "${WORK}/expected-updated/${f}")
endforeach()
file(COPY "${WORK}/expected-update/" DESTINATION "${WORK}/expected-updated")
run(diff -r "${WORK}/expected-updated" "${WORK}/game/CUSA12776-v01.05")
run(diff -r "${WORK}/expected" "${WORK}/game/CUSA12776")
run(diff -r "${WORK}/expected-dlc" "${WORK}/game/CUSA12776-DLC/HOKUTOGADLC00001")
# From an unpacked folder, into a second game folder.
run("${LPN_PKG}" extract "${WORK}/test.pkg" "${WORK}/game2")
run("${LPN_PKG}" extract "${WORK}/update.pkg" "${WORK}/unpacked")
run("${LPN_PKG}" apply "${WORK}/unpacked/CUSA12776-UPDATE" "${WORK}/game2")
run(diff -r "${WORK}/expected-updated" "${WORK}/game2/CUSA12776-v01.05")
run(diff -r "${WORK}/expected-update" "${WORK}/unpacked/CUSA12776-UPDATE")
