# GUI Framework Analysis for Polyglott

## Executive Summary

This document analyzes GUI framework options for Polyglott, a voice-first language tutor for children aged 3-12. The primary requirements are:

- **Visually appealing** - Child-friendly, modern interface
- **Compact packaging** - Reasonable app size for distribution
- **macOS primary target** - Apple Silicon optimization
- **Audio integration** - Microphone/speaker support
- **Cross-platform potential** - Windows/Linux secondary targets

## Framework Comparison

### 1. Flet (Flutter-based)

| Aspect | Details |
|--------|---------|
| **Technology** | Python wrapper around Flutter |
| **Platforms** | Windows, macOS, Linux, Web, iOS, Android |
| **Learning Curve** | Low - simple, declarative API |
| **Package Size** | 77-100+ MB (includes Flutter runtime) |
| **Look & Feel** | Modern Material Design, very polished |

#### Pros
- **Fastest development time** - Simple, intuitive API similar to Flutter
- **Modern appearance** - Material Design out-of-the-box, perfect for children
- **Hot reload** - Rapid iteration during development
- **True cross-platform** - Same codebase for desktop, mobile, and web
- **Active development** - Version 0.80.0 (Jan 2025), rapid improvements
- **No frontend experience required** - Pure Python

#### Cons
- **Larger package size** - 77-100+ MB due to Flutter runtime and libmpv
- **Newer framework** - Less mature than Qt alternatives
- **Flutter dependency** - Requires Flutter SDK for `flet build`
- **Limited native look** - Always looks like Flutter, not native OS

#### Packaging
```bash
# Option 1: PyInstaller (simpler, ~77MB)
pip install pyinstaller
flet pack main.py

# Option 2: Flutter SDK (more control, 100+ MB)
flet build macos
```

#### Best For
Internal tools, prototypes, modern dashboards, apps where Material Design is desirable.

---

### 2. PySide6 (Qt6)

| Aspect | Details |
|--------|---------|
| **Technology** | Official Qt6 Python bindings |
| **Platforms** | Windows, macOS, Linux, embedded |
| **Learning Curve** | Medium-High - extensive API |
| **Package Size** | 50-150 MB (depends on modules used) |
| **Look & Feel** | Native or fully customizable |

#### Pros
- **Industry standard** - Qt is battle-tested, used in production everywhere
- **Most powerful** - Every widget imaginable, multimedia, 3D, WebEngine
- **Native appearance** - Can look native on each OS
- **Qt Quick/QML** - Modern declarative UI option for touch interfaces
- **Excellent documentation** - Comprehensive and professional
- **LGPL license** - Can be used in closed-source apps

#### Cons
- **Steep learning curve** - Qt is vast and complex
- **Larger package size** - QtWebEngine alone adds 50+ MB
- **Verbose code** - More boilerplate than Flet
- **Overkill for simple apps** - Lots of unused functionality

#### Package Size Optimization
```python
# Use pyside6-deploy to exclude unused modules
# Excludes: QtWebEngine, QtQuick3D, QtCharts, QtTest, QtSensors
# Can reduce size by 30-50%
```

#### Best For
Professional applications, complex UIs, apps requiring native look, long-term maintainability.

---

### 3. Kivy (OpenGL-based)

| Aspect | Details |
|--------|---------|
| **Technology** | OpenGL ES 2 rendering |
| **Platforms** | Windows, macOS, Linux, Android, iOS, Raspberry Pi |
| **Learning Curve** | Medium - unique concepts |
| **Package Size** | 30-60 MB |
| **Look & Feel** | Custom/Modern, touch-optimized |

#### Pros
- **Touch-first design** - Perfect for children's apps
- **Smooth animations** - GPU-accelerated, 60 FPS
- **Mobile support** - Buildozer for Android/iOS
- **KivyMD** - Material Design extension available
- **Smaller packages** - More compact than Qt/Flet
- **Unique aesthetic** - Stands out, game-like feel

#### Cons
- **Non-native look** - Always looks like Kivy
- **Smaller community** - Less Stack Overflow help than Qt
- **KV language** - Separate DSL to learn for layouts
- **Buildozer complexity** - Mobile packaging is tricky

#### Best For
Touch applications, mobile-first apps, children's games, educational apps.

---

### 4. DearPyGui (Dear ImGui)

| Aspect | Details |
|--------|---------|
| **Technology** | GPU-rendered, Dear ImGui-based |
| **Platforms** | Windows, macOS, Linux |
| **Learning Curve** | Low-Medium |
| **Package Size** | 15-25 MB (lightest option) |
| **Look & Feel** | Technical/Dashboard style |

#### Pros
- **Extremely lightweight** - Minimal dependencies
- **Fastest rendering** - 60+ FPS, C++ backend
- **Best for data visualization** - Built-in plots, real-time updates
- **Smallest package size** - 15-25 MB
- **Simple API** - Retained mode, easy to learn

#### Cons
- **Technical aesthetic** - Not child-friendly out-of-box
- **No mobile support** - Desktop only
- **Limited widgets** - Fewer than Qt
- **Gaming/tool aesthetic** - May feel too "serious"

#### Best For
Scientific apps, data dashboards, developer tools, real-time visualizations.

---

### 5. NiceGUI (Web-based)

| Aspect | Details |
|--------|---------|
| **Technology** | FastAPI + Vue.js + Tailwind CSS |
| **Platforms** | Web (runs locally or deployed) |
| **Learning Curve** | Low |
| **Package Size** | N/A (web-based) or Docker container |
| **Look & Feel** | Modern web UI |

#### Pros
- **Beautiful web UI** - Tailwind CSS styling
- **Easy to learn** - Streamlit-like simplicity
- **Reactive updates** - Real-time WebSocket communication
- **Docker deployment** - Easy cloud deployment

#### Cons
- **Not a desktop app** - Runs in browser
- **Requires browser** - Not a native experience
- **Network dependency** - WebSocket-based
- **Audio complexity** - Browser audio API limitations

#### Best For
Dashboards, internal tools, web apps, not ideal for polyglott's use case.

---

## Package Size Comparison

| Framework | Minimal Package | Typical Package | Notes |
|-----------|----------------|-----------------|-------|
| **DearPyGui** | ~15 MB | ~25 MB | Lightest option |
| **Kivy** | ~30 MB | ~60 MB | Depends on modules |
| **PySide6** | ~50 MB | ~150 MB | Excludable modules help |
| **Flet** | ~77 MB | ~100+ MB | Flutter runtime overhead |

*Note: All sizes approximate and depend on dependencies, compression, and packaging method.*

---

## Recommendation for Polyglott

### Primary Recommendation: **Flet**

Given Polyglott's requirements:

1. **Child-friendly UI** - Flet's Material Design is colorful, modern, and appealing to children
2. **Rapid development** - Simple API means faster iteration
3. **Audio support** - Built-in audio/video player support
4. **Cross-platform** - Future mobile app potential for tablets (common for kids)
5. **Active development** - Frequent updates and improvements

**Package size trade-off**: 77-100 MB is acceptable for a desktop app with embedded models (ML models will dwarf UI framework size anyway).

### Secondary Recommendation: **Kivy + KivyMD**

If package size is critical or you want a more unique, game-like interface:

1. **Touch-optimized** - Great for children's interactions
2. **Smaller packages** - 30-60 MB
3. **Animation support** - Engaging for kids
4. **Mobile potential** - Android/iOS via Buildozer

### Alternative: **PySide6**

If you need maximum control, native look, or long-term maintainability:

1. **Qt Quick/QML** - Modern touch interfaces
2. **Multimedia support** - QtMultimedia for audio
3. **Professional quality** - Enterprise-grade framework
4. **Larger learning curve** - More investment required

---

## Implementation Considerations

### Audio Integration

All frameworks can work with Polyglott's existing audio stack:

```python
# Polyglott uses:
# - sounddevice: Recording (VAD, STT)
# - soundfile: Audio file handling
# - Custom AudioPlayer: Playback (TTS output)

# These are independent of UI framework
# GUI only needs to:
# 1. Display state (Listening, Processing, Speaking)
# 2. Show conversation history
# 3. Handle user profile selection
# 4. Provide settings UI
```

### Suggested Architecture

```
┌─────────────────────────────────────────┐
│              GUI Layer (Flet)           │
│  ┌─────────┐ ┌─────────┐ ┌───────────┐  │
│  │ Profile │ │  Chat   │ │ Settings  │  │
│  │ Select  │ │ Display │ │   Panel   │  │
│  └─────────┘ └─────────┘ └───────────┘  │
└──────────────────┬──────────────────────┘
                   │ Events/Callbacks
┌──────────────────▼──────────────────────┐
│      ConversationManager (Existing)     │
│  ┌────────┐ ┌─────┐ ┌─────┐ ┌────────┐  │
│  │  VAD   │ │ STT │ │ LLM │ │  TTS   │  │
│  └────────┘ └─────┘ └─────┘ └────────┘  │
└─────────────────────────────────────────┘
```

### Child-Friendly UI Elements

For a children's language tutor, consider:

- **Large buttons** - Easy to tap/click
- **Bright colors** - Engaging and fun
- **Character/mascot** - Visual tutor representation
- **Progress indicators** - Gamification elements
- **Simple navigation** - Minimal complexity
- **Voice guidance** - Already implemented in CLI

---

## Next Steps

1. **Prototype with Flet** - Create basic UI with profile selection, chat display, and listening indicator
2. **Integrate audio callbacks** - Connect GUI state to ConversationManager events
3. **Test packaging** - Build standalone app with `flet build macos`
4. **Evaluate size** - Measure actual package size with all dependencies
5. **User testing** - Get feedback from target age group

---

## References

- [Flet Official](https://flet.dev/)
- [Flet Packaging](https://flet.dev/docs/publish/)
- [PySide6 Documentation](https://doc.qt.io/qtforpython-6/)
- [PySide6 Deployment](https://doc.qt.io/qtforpython-6/deployment/deployment-pyside6-deploy.html)
- [Kivy Framework](https://kivy.org/)
- [Kivy Packaging](https://kivy.org/doc/stable/guide/packaging.html)
- [DearPyGui](https://github.com/hoffstadt/DearPyGui)
- [Python GUIs Comparison](https://www.pythonguis.com/faq/which-python-gui-library/)
- [GeeksforGeeks GUI Frameworks 2025](https://www.geeksforgeeks.org/blogs/best-python-gui-frameworks-for-developers/)
