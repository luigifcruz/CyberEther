#ifndef JETSTREAM_RUNTIME_HELPERS_HH
#define JETSTREAM_RUNTIME_HELPERS_HH

#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "jetstream/logger.hh"
#include "jetstream/types.hh"

namespace Jetstream {

inline std::string IndentSourcePiece(const std::string& piece, const std::string& indentation) {
    std::string indentedPiece;

    std::size_t lineStart = 0;
    while (lineStart < piece.size()) {
        const std::size_t lineEnd = piece.find('\n', lineStart);
        const bool hasNewline = lineEnd != std::string::npos;

        std::string line = hasNewline ? piece.substr(lineStart, lineEnd - lineStart)
                                      : piece.substr(lineStart);

        const bool hasCarriageReturn = !line.empty() && line.back() == '\r';
        if (hasCarriageReturn) {
            line.pop_back();
        }

        if (!line.empty()) {
            indentedPiece += indentation;
        }

        indentedPiece += line;

        if (!hasNewline) {
            break;
        }

        indentedPiece += hasCarriageReturn ? "\r\n" : "\n";
        lineStart = lineEnd + 1;
    }

    return indentedPiece;
}

inline Result ExpandSourcePieces(const std::string& sourceTemplate,
                                 const std::unordered_map<std::string, std::string>& pieces,
                                 std::string& expandedSource) {
    expandedSource.clear();

    std::unordered_set<std::string> usedPieces;

    std::size_t lineStart = 0;
    while (lineStart < sourceTemplate.size()) {
        const std::size_t lineEnd = sourceTemplate.find('\n', lineStart);
        const bool hasNewline = lineEnd != std::string::npos;

        const std::string rawLine = hasNewline ? sourceTemplate.substr(lineStart, lineEnd - lineStart)
                                               : sourceTemplate.substr(lineStart);

        std::string line = rawLine;
        const bool hasCarriageReturn = !line.empty() && line.back() == '\r';
        if (hasCarriageReturn) {
            line.pop_back();
        }

        std::size_t contentStart = 0;
        while (contentStart < line.size() && (line[contentStart] == ' ' || line[contentStart] == '\t')) {
            contentStart++;
        }

        const std::string indentation = line.substr(0, contentStart);

        static constexpr const char* markerPrefix = "<<<";
        static constexpr const char* markerSuffix = ">>>";

        if (line.compare(contentStart, std::char_traits<char>::length(markerPrefix), markerPrefix) == 0) {
            const std::size_t pieceStart = contentStart + std::char_traits<char>::length(markerPrefix);
            const std::size_t pieceEnd = line.find(markerSuffix, pieceStart);
            const bool isPlaceholder = pieceEnd != std::string::npos &&
                                       line.find_first_not_of(" \t",
                                                              pieceEnd + std::char_traits<char>::length(markerSuffix)) == std::string::npos;

            if (isPlaceholder) {
                const std::string pieceId = line.substr(pieceStart, pieceEnd - pieceStart);
                const auto piece = pieces.find(pieceId);

                if (piece == pieces.end()) {
                    JST_ERROR("[RUNTIME] Source references missing piece '{}'.", pieceId);
                    return Result::ERROR;
                }

                usedPieces.insert(pieceId);

                const std::string indentedPiece = IndentSourcePiece(piece->second, indentation);
                expandedSource += indentedPiece;

                if (hasNewline && (indentedPiece.empty() || indentedPiece.back() != '\n')) {
                    expandedSource += hasCarriageReturn ? "\r\n" : "\n";
                }

                lineStart = hasNewline ? lineEnd + 1 : sourceTemplate.size();
                continue;
            }
        }

        expandedSource += rawLine;

        if (hasNewline) {
            expandedSource += '\n';
        }

        lineStart = hasNewline ? lineEnd + 1 : sourceTemplate.size();
    }

    for (const auto& piece : pieces) {
        if (!usedPieces.contains(piece.first)) {
            JST_ERROR("[RUNTIME] Piece '{}' was provided but not used in the source template.", piece.first);
            return Result::ERROR;
        }
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream

#endif  // JETSTREAM_RUNTIME_HELPERS_HH
