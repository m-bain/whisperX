"""
WhisperX MCP Server - Model Context Protocol integration

This MCP server exposes WhisperX transcriptions as context resources
for Claude Desktop and other MCP-compatible applications.

Features:
- List all transcriptions
- Get transcript content with timestamps and speakers
- Search transcripts semantically (via Gemini if available)
- Query transcript metadata

Usage:
    python mcp_server.py

Claude Desktop configuration:
    Add to ~/Library/Application Support/Claude/claude_desktop_config.json:
    {
      "mcpServers": {
        "whisperx": {
          "command": "python",
          "args": ["/path/to/whisperX/mcp_server.py"]
        }
      }
    }
"""

import json
import sys
import os
from typing import Any, Sequence
from datetime import datetime

# MCP SDK imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource
    from pydantic import AnyUrl
except ImportError:
    print("Error: MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Supabase for database access
try:
    from supabase import create_client, Client
except ImportError:
    print("Error: Supabase not installed. Install with: pip install supabase", file=sys.stderr)
    sys.exit(1)

# Environment variables
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables required", file=sys.stderr)
    sys.exit(1)

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Create MCP server
app = Server("whisperx-transcriptions")


@app.list_resources()
async def list_resources() -> list[Resource]:
    """List all available transcription resources"""
    try:
        # Fetch all transcriptions from Supabase
        response = supabase.table("transcripts").select("*").eq("status", "completed").order("createdAt", desc=True).execute()

        resources = []
        for transcript in response.data:
            resources.append(Resource(
                uri=AnyUrl(f"whisperx://transcript/{transcript['id']}"),
                name=f"Transcript: {transcript['fileName']}",
                mimeType="text/plain",
                description=f"Transcription of {transcript['fileName']} ({transcript.get('language', 'unknown')} language, "
                           f"{transcript.get('durationSeconds', 0)}s duration, "
                           f"{transcript.get('speakers', {}).get('count', 0)} speakers)"
            ))

        return resources
    except Exception as e:
        print(f"Error listing resources: {e}", file=sys.stderr)
        return []


@app.read_resource()
async def read_resource(uri: AnyUrl) -> str:
    """Read the content of a specific transcription resource"""
    try:
        # Extract transcript ID from URI: whisperx://transcript/{id}
        transcript_id = str(uri).split("/")[-1]

        # Fetch transcript from Supabase
        response = supabase.table("transcripts").select("*").eq("id", transcript_id).single().execute()
        transcript = response.data

        if not transcript:
            return "Error: Transcript not found"

        # Format transcript content with metadata and segments
        content_parts = [
            f"# Transcript: {transcript['fileName']}",
            f"",
            f"**Language:** {transcript.get('language', 'unknown')}",
            f"**Duration:** {transcript.get('durationSeconds', 0)} seconds ({transcript.get('durationSeconds', 0) / 60:.1f} minutes)",
            f"**Created:** {transcript.get('createdAt', 'unknown')}",
            f"**Status:** {transcript.get('status', 'unknown')}",
        ]

        # Add speaker info if available
        speakers_data = transcript.get('speakers')
        if speakers_data:
            content_parts.append(f"**Speakers:** {speakers_data.get('count', 0)} ({', '.join(speakers_data.get('labels', []))})")

        content_parts.append("")
        content_parts.append("## Transcription")
        content_parts.append("")

        # Add segments with timestamps and speakers
        segments = transcript.get('segments', [])
        if segments:
            for i, segment in enumerate(segments):
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                text = segment.get('text', '')
                speaker = segment.get('speaker')

                timestamp = f"[{start:.2f}s - {end:.2f}s]"
                speaker_label = f"**{speaker}:** " if speaker else ""
                content_parts.append(f"{timestamp} {speaker_label}{text}")
        else:
            # Fallback to full text if no segments
            content_parts.append(transcript.get('transcriptText', 'No transcript available'))

        return "\n".join(content_parts)

    except Exception as e:
        print(f"Error reading resource: {e}", file=sys.stderr)
        return f"Error: {str(e)}"


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for interacting with transcriptions"""
    return [
        Tool(
            name="search_transcripts",
            description="Search through all transcriptions for specific content or keywords",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (keyword or phrase to find in transcriptions)"
                    },
                    "language": {
                        "type": "string",
                        "description": "Optional: Filter by language code (e.g., 'en', 'it', 'fr')"
                    },
                    "min_duration": {
                        "type": "number",
                        "description": "Optional: Minimum duration in seconds"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_transcript_stats",
            description="Get statistics and metadata for all transcriptions",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_failed": {
                        "type": "boolean",
                        "description": "Include failed transcriptions in stats (default: false)"
                    }
                }
            }
        ),
        Tool(
            name="get_speakers",
            description="List all unique speakers across transcriptions with their transcript counts",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool calls"""
    try:
        if name == "search_transcripts":
            query = arguments.get("query", "").lower()
            language = arguments.get("language")
            min_duration = arguments.get("min_duration")

            # Build query
            db_query = supabase.table("transcripts").select("*").eq("status", "completed")

            if language:
                db_query = db_query.eq("language", language)
            if min_duration:
                db_query = db_query.gte("durationSeconds", min_duration)

            response = db_query.execute()

            # Filter by text content
            results = []
            for transcript in response.data:
                transcript_text = transcript.get('transcriptText', '').lower()
                if query in transcript_text:
                    results.append({
                        "id": transcript['id'],
                        "fileName": transcript['fileName'],
                        "language": transcript.get('language'),
                        "duration": transcript.get('durationSeconds'),
                        "created": transcript.get('createdAt'),
                        "preview": transcript_text[:200] + "..." if len(transcript_text) > 200 else transcript_text
                    })

            result_text = f"Found {len(results)} transcription(s) matching '{query}':\n\n"
            for r in results:
                result_text += f"- **{r['fileName']}** ({r['language']}, {r['duration']}s)\n"
                result_text += f"  ID: {r['id']}\n"
                result_text += f"  Preview: {r['preview']}\n\n"

            return [TextContent(type="text", text=result_text)]

        elif name == "get_transcript_stats":
            include_failed = arguments.get("include_failed", False)

            # Build query
            db_query = supabase.table("transcripts").select("*")
            if not include_failed:
                db_query = db_query.eq("status", "completed")

            response = db_query.execute()
            transcripts = response.data

            # Calculate stats
            total_count = len(transcripts)
            total_duration = sum(t.get('durationSeconds', 0) for t in transcripts)
            languages = {}
            speaker_counts = {}

            for t in transcripts:
                lang = t.get('language', 'unknown')
                languages[lang] = languages.get(lang, 0) + 1

                speakers_data = t.get('speakers')
                if speakers_data:
                    count = speakers_data.get('count', 0)
                    speaker_counts[count] = speaker_counts.get(count, 0) + 1

            stats_text = f"# Transcription Statistics\n\n"
            stats_text += f"**Total transcriptions:** {total_count}\n"
            stats_text += f"**Total duration:** {total_duration}s ({total_duration / 60:.1f} minutes)\n"
            stats_text += f"**Average duration:** {total_duration / total_count if total_count > 0 else 0:.1f}s\n\n"
            stats_text += f"**Languages:**\n"
            for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
                stats_text += f"  - {lang}: {count}\n"
            stats_text += f"\n**Speaker distribution:**\n"
            for count, freq in sorted(speaker_counts.items()):
                stats_text += f"  - {count} speaker(s): {freq} transcription(s)\n"

            return [TextContent(type="text", text=stats_text)]

        elif name == "get_speakers":
            response = supabase.table("transcripts").select("*").eq("status", "completed").execute()

            speaker_transcripts = {}
            for transcript in response.data:
                speakers_data = transcript.get('speakers')
                if speakers_data:
                    for speaker in speakers_data.get('labels', []):
                        if speaker not in speaker_transcripts:
                            speaker_transcripts[speaker] = []
                        speaker_transcripts[speaker].append(transcript['fileName'])

            result_text = f"# Speakers Across Transcriptions\n\n"
            result_text += f"Found {len(speaker_transcripts)} unique speaker label(s):\n\n"

            for speaker, files in sorted(speaker_transcripts.items()):
                result_text += f"**{speaker}** (appears in {len(files)} file(s)):\n"
                for file in files[:5]:  # Show first 5 files
                    result_text += f"  - {file}\n"
                if len(files) > 5:
                    result_text += f"  ... and {len(files) - 5} more\n"
                result_text += "\n"

            return [TextContent(type="text", text=result_text)]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        print(f"Error calling tool {name}: {e}", file=sys.stderr)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run the MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
