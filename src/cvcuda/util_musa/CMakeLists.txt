# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

find_package(MUSAToolkit REQUIRED)

add_library(cvcuda_util STATIC
    Event.cpp
    Stream.cpp
    StreamId.cpp
)

target_compile_definitions(cvcuda_util PUBLIC NVCV_USE_MUSA=1)

target_link_libraries(cvcuda_util
    PUBLIC
        nvcv_util
        MUSA::musart_static
        -lrt
)
