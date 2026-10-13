#include "gcn/shader.h"

#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include <mutex>

namespace gcn {

std::vector<uint32_t> compileGlsl(const std::string& source, GlslStage stage, std::string* error) {
    static std::once_flag once;
    std::call_once(once, [] { glslang::InitializeProcess(); });

    EShLanguage lang = EShLangVertex;
    switch (stage) {
    case GlslStage::Vertex: lang = EShLangVertex; break;
    case GlslStage::Geometry: lang = EShLangGeometry; break;
    case GlslStage::Fragment: lang = EShLangFragment; break;
    case GlslStage::Compute: lang = EShLangCompute; break;
    }
    glslang::TShader shader(lang);
    const char* text = source.c_str();
    shader.setStrings(&text, 1);
    shader.setEnvInput(glslang::EShSourceGlsl, lang, glslang::EShClientVulkan, 100);
    shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_2);
    shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_5);
    const auto messages = EShMessages(EShMsgSpvRules | EShMsgVulkanRules);
    if (!shader.parse(GetDefaultResources(), 450, false, messages)) {
        if (error) *error = shader.getInfoLog();
        return {};
    }
    glslang::TProgram program;
    program.addShader(&shader);
    if (!program.link(messages)) {
        if (error) *error = program.getInfoLog();
        return {};
    }
    std::vector<uint32_t> spirv;
    glslang::SpvOptions options;
    options.disableOptimizer = false;
    options.optimizeSize = false;
    glslang::GlslangToSpv(*program.getIntermediate(lang), spirv, &options);
    return spirv;
}

} // namespace gcn
