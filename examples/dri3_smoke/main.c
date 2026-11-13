/* DRI3+X11 end-to-end smoke for vantage:
 *  1. Create an override-redirect window on $DISPLAY via Xlib.
 *  2. eglCreateWindowSurface + GLES1 render (solid color + triangle).
 *  3. eglSwapBuffers -> DRI3/Present flip (or MIT-SHM fallback).
 *  4. Server-side readback of the window via XGetImage proves the
 *     presented pixmap reached the screen.
 * Compile: cc dri3_smoke.c -lX11 -lEGL -lGLESv1_CM   (see build.sh)
 */
#define GL_GLEXT_PROTOTYPES 1
#define EGL_EGLEXT_PROTOTYPES 1
#include <EGL/egl.h>
#include <GLES/gl.h>
#define XUTIL_DEFINE_FUNCTIONS
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void die(const char *what) {
    fprintf(stderr, "SMOKE FAIL: %s (egl err 0x%x)\n", what, eglGetError());
    exit(1);
}

int main(int argc, char **argv) {
    Display *dpy = XOpenDisplay(NULL);
    if (!dpy) { fprintf(stderr, "SMOKE FAIL: XOpenDisplay\n"); return 1; }
    int scr = DefaultScreen(dpy);
    Window root = RootWindow(dpy, scr);

    const int W = 128, H = 96;
    XSetWindowAttributes swa = {0};
    swa.override_redirect = True;
    swa.event_mask = ExposureMask;
    Window win = XCreateWindow(dpy, root, 0, 0, W, H, 0, CopyFromParent,
                               InputOutput, CopyFromParent, CWOverrideRedirect, &swa);
    XMapWindow(dpy, win);
    XFlush(dpy);

    EGLDisplay edpy = eglGetDisplay((EGLNativeDisplayType)dpy);
    if (edpy == EGL_NO_DISPLAY) die("eglGetDisplay");
    EGLint maj, min;
    if (!eglInitialize(edpy, &maj, &min)) die("eglInitialize");
    printf("EGL %d.%d vendor: %s\n", maj, min, eglQueryString(edpy, EGL_VENDOR));
    if (!eglBindAPI(EGL_OPENGL_ES_API)) die("eglBindAPI");

    EGLConfig cfg;
    EGLint ncfg = 0;
    const EGLint cfgattr[] = { EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
                               EGL_SURFACE_TYPE, EGL_WINDOW_BIT, EGL_NONE };
    if (!eglChooseConfig(edpy, cfgattr, &cfg, 1, &ncfg) || ncfg < 1) die("eglChooseConfig");

    EGLSurface surf = eglCreateWindowSurface(edpy, cfg, (EGLNativeWindowType)win, NULL);
    if (surf == EGL_NO_SURFACE) die("eglCreateWindowSurface");
    EGLContext ctx = eglCreateContext(edpy, cfg, EGL_NO_CONTEXT, NULL);
    if (ctx == EGL_NO_CONTEXT) die("eglCreateContext");
    if (!eglMakeCurrent(edpy, surf, surf, ctx)) die("eglMakeCurrent");

    printf("GL_RENDERER=%s GL_VENDOR=%s\n", glGetString(GL_RENDERER), glGetString(GL_VENDOR));

    // Frame 1: solid magenta.
    glViewport(0, 0, W, H);
    glClearColor(1.0f, 0.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    if (!eglSwapBuffers(edpy, surf)) die("eglSwapBuffers (magenta)");
    XSync(dpy, False);
    usleep(150000); // Present idle path completes; server composites.

    // Server-side readback of the window center.
    XImage *img = XGetImage(dpy, win, W/2-2, H/2-2, 4, 4, AllPlanes, ZPixmap);
    if (!img) die("XGetImage (magenta)");
    unsigned long px = XGetPixel(img, 2, 2);
    XDestroyImage(img);
    printf("magenta frame center: 0x%08lx\n", px);
    int r = (px >> 0) & 0xff, g = (px >> 8) & 0xff, b = (px >> 16) & 0xff;
    int magenta_ok = r > 200 && g < 60 && b > 200;

    // Frame 2: triangle over cyan, drawn twice to exercise buffer reuse.
    glClearColor(0.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
    static const GLfloat verts[6] = { -0.8f, -0.8f, 0.8f, -0.8f, 0.0f, 0.8f };
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, 0, verts);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisableClientState(GL_VERTEX_ARRAY);
    if (!eglSwapBuffers(edpy, surf)) die("eglSwapBuffers (cyan)");
    if (!eglSwapBuffers(edpy, surf)) die("eglSwapBuffers (reuse)");
    XSync(dpy, False);
    usleep(150000);

    img = XGetImage(dpy, win, W/2-2, H/2-2, 4, 4, AllPlanes, ZPixmap);
    if (!img) die("XGetImage (cyan)");
    px = XGetPixel(img, 2, 2);
    XDestroyImage(img);
    printf("cyan frame center: 0x%08lx\n", px);
    int cyan_ok = ((px >> 8) & 0xff) > 200 && ((px >> 0) & 0xff) < 60;

    int pass = magenta_ok && cyan_ok;
    printf("SMOKE %s (magenta=%d cyan=%d)\n", pass ? "PASS" : "FAIL", magenta_ok, cyan_ok);

    XDestroyWindow(dpy, win);
    XCloseDisplay(dpy);
    return pass ? 0 : 1;
}
