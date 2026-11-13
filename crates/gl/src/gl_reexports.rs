// Generated GLES C ABI re-exports
use core::ffi::*;
use volt_gles::*;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glMatrixMode(mode: GLenum) {
    volt_gles::glMatrixMode(mode)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glLoadIdentity() {
    volt_gles::glLoadIdentity()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glPushMatrix() {
    volt_gles::glPushMatrix()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glPopMatrix() {
    volt_gles::glPopMatrix()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glTranslatef(x: GLfloat, y: GLfloat, z: GLfloat) {
    volt_gles::glTranslatef(x, y, z)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glTranslated(x: GLdouble, y: GLdouble, z: GLdouble) {
    volt_gles::glTranslated(x, y, z)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glRotatef(angle: GLfloat, x: GLfloat, y: GLfloat, z: GLfloat) {
    volt_gles::glRotatef(angle, x, y, z)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glScalef(x: GLfloat, y: GLfloat, z: GLfloat) {
    volt_gles::glScalef(x, y, z)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glScaled(x: GLdouble, y: GLdouble, z: GLdouble) {
    volt_gles::glScaled(x, y, z)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glOrtho(
    left: GLdouble,
    right: GLdouble,
    bottom: GLdouble,
    top: GLdouble,
    near_val: GLdouble,
    far_val: GLdouble,
) {
    volt_gles::glOrtho(left, right, bottom, top, near_val, far_val)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glOrthof(
    left: GLfloat,
    right: GLfloat,
    bottom: GLfloat,
    top: GLfloat,
    near_val: GLfloat,
    far_val: GLfloat,
) {
    volt_gles::glOrthof(left, right, bottom, top, near_val, far_val)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glFrustum(
    left: GLdouble,
    right: GLdouble,
    bottom: GLdouble,
    top: GLdouble,
    near_val: GLdouble,
    far_val: GLdouble,
) {
    volt_gles::glFrustum(left, right, bottom, top, near_val, far_val)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glFrustumf(
    left: GLfloat,
    right: GLfloat,
    bottom: GLfloat,
    top: GLfloat,
    near_val: GLfloat,
    far_val: GLfloat,
) {
    volt_gles::glFrustumf(left, right, bottom, top, near_val, far_val)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glMultMatrixf(m: *const GLfloat) {
    volt_gles::glMultMatrixf(m)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glLoadMatrixf(m: *const GLfloat) {
    volt_gles::glLoadMatrixf(m)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glEnableClientState(array: GLenum) {
    volt_gles::glEnableClientState(array)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glDisableClientState(array: GLenum) {
    volt_gles::glDisableClientState(array)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glVertexPointer(
    size: GLint,
    type_: GLenum,
    stride: GLsizei,
    pointer: *const c_void,
) {
    volt_gles::glVertexPointer(size, type_, stride, pointer)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glTexCoordPointer(
    size: GLint,
    type_: GLenum,
    stride: GLsizei,
    pointer: *const c_void,
) {
    volt_gles::glTexCoordPointer(size, type_, stride, pointer)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glColorPointer(
    size: GLint,
    type_: GLenum,
    stride: GLsizei,
    pointer: *const c_void,
) {
    volt_gles::glColorPointer(size, type_, stride, pointer)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glNormalPointer(type_: GLenum, stride: GLsizei, pointer: *const c_void) {
    volt_gles::glNormalPointer(type_, stride, pointer)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glClientActiveTexture(texture: GLenum) {
    volt_gles::glClientActiveTexture(texture)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glDrawArrays(mode: GLenum, first: GLint, count: GLsizei) {
    volt_gles::glDrawArrays(mode, first, count)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glDrawElements(
    mode: GLenum,
    count: GLsizei,
    type_: GLenum,
    indices: *const c_void,
) {
    volt_gles::glDrawElements(mode, count, type_, indices)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glBegin(mode: GLenum) {
    volt_gles::glBegin(mode)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glEnd() {
    volt_gles::glEnd()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glVertex3f(x: GLfloat, y: GLfloat, z: GLfloat) {
    volt_gles::glVertex3f(x, y, z)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glVertex2f(x: GLfloat, y: GLfloat) {
    volt_gles::glVertex2f(x, y)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glTexCoord2f(u: GLfloat, v: GLfloat) {
    volt_gles::glTexCoord2f(u, v)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glMultiTexCoord2f(target: GLenum, s: GLfloat, t: GLfloat) {
    volt_gles::glMultiTexCoord2f(target, s, t)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glMultiTexCoord4f(
    target: GLenum,
    s: GLfloat,
    t: GLfloat,
    r: GLfloat,
    q: GLfloat,
) {
    volt_gles::glMultiTexCoord4f(target, s, t, r, q)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glColor4f(r: GLfloat, g: GLfloat, b: GLfloat, a: GLfloat) {
    volt_gles::glColor4f(r, g, b, a)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glColor3f(r: GLfloat, g: GLfloat, b: GLfloat) {
    volt_gles::glColor3f(r, g, b)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glColor4ub(r: GLubyte, g: GLubyte, b: GLubyte, a: GLubyte) {
    volt_gles::glColor4ub(r, g, b, a)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glColor3ub(r: GLubyte, g: GLubyte, b: GLubyte) {
    volt_gles::glColor3ub(r, g, b)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glColor4fv(v: *const GLfloat) {
    volt_gles::glColor4fv(v)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glNormal3f(x: GLfloat, y: GLfloat, z: GLfloat) {
    volt_gles::glNormal3f(x, y, z)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glGenLists(range: GLsizei) -> GLuint {
    volt_gles::glGenLists(range)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glDeleteLists(list: GLuint, range: GLsizei) {
    volt_gles::glDeleteLists(list, range)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glNewList(list: GLuint, mode: GLenum) {
    volt_gles::glNewList(list, mode)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glEndList() {
    volt_gles::glEndList()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glCallList(list: GLuint) {
    volt_gles::glCallList(list)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glCallLists(n: GLsizei, type_: GLenum, lists: *const c_void) {
    volt_gles::glCallLists(n, type_, lists)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glIsList(list: GLuint) -> GLboolean {
    volt_gles::glIsList(list)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glGenTextures(n: GLsizei, textures: *mut GLuint) {
    volt_gles::glGenTextures(n, textures)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glDeleteTextures(n: GLsizei, textures: *const GLuint) {
    volt_gles::glDeleteTextures(n, textures)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glBindTexture(target: GLenum, texture: GLuint) {
    volt_gles::glBindTexture(target, texture)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glTexImage2D(
    target: GLenum,
    level: GLint,
    internalformat: GLint,
    width: GLsizei,
    height: GLsizei,
    border: GLint,
    format: GLenum,
    type_: GLenum,
    pixels: *const c_void,
) {
    volt_gles::glTexImage2D(
        target,
        level,
        internalformat,
        width,
        height,
        border,
        format,
        type_,
        pixels,
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glTexSubImage2D(
    target: GLenum,
    level: GLint,
    xoffset: GLint,
    yoffset: GLint,
    width: GLsizei,
    height: GLsizei,
    format: GLenum,
    type_: GLenum,
    pixels: *const c_void,
) {
    volt_gles::glTexSubImage2D(
        target, level, xoffset, yoffset, width, height, format, type_, pixels,
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glTexParameteri(target: GLenum, pname: GLenum, param: GLint) {
    volt_gles::glTexParameteri(target, pname, param)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glTexParameterf(target: GLenum, pname: GLenum, param: GLfloat) {
    volt_gles::glTexParameterf(target, pname, param)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glTexParameteriv(target: GLenum, pname: GLenum, params: *const GLint) {
    volt_gles::glTexParameteriv(target, pname, params)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glTexParameterfv(target: GLenum, pname: GLenum, params: *const GLfloat) {
    volt_gles::glTexParameterfv(target, pname, params)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glActiveTexture(texture: GLenum) {
    volt_gles::glActiveTexture(texture)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glTexImage1D(
    target: GLenum,
    level: GLint,
    internalformat: GLint,
    width: GLsizei,
    border: GLint,
    format: GLenum,
    type_: GLenum,
    pixels: *const c_void,
) {
    volt_gles::glTexImage1D(
        target,
        level,
        internalformat,
        width,
        border,
        format,
        type_,
        pixels,
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glTexImage3D(
    target: GLenum,
    level: GLint,
    internalformat: GLint,
    width: GLsizei,
    height: GLsizei,
    depth: GLsizei,
    border: GLint,
    format: GLenum,
    type_: GLenum,
    pixels: *const c_void,
) {
    volt_gles::glTexImage3D(
        target,
        level,
        internalformat,
        width,
        height,
        depth,
        border,
        format,
        type_,
        pixels,
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glGetTexLevelParameteri(
    target: GLenum,
    level: GLint,
    pname: GLenum,
    params: *mut GLint,
) {
    volt_gles::glGetTexLevelParameteri(target, level, pname, params)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glTexGen(coord: GLenum, pname: GLenum, param: GLfloat) {
    volt_gles::glTexGen(coord, pname, param)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glTexGeni(coord: GLenum, pname: GLenum, param: GLint) {
    volt_gles::glTexGeni(coord, pname, param)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glTexEnvf(target: GLenum, pname: GLenum, param: GLfloat) {
    volt_gles::glTexEnvf(target, pname, param)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glTexEnvi(target: GLenum, pname: GLenum, param: GLint) {
    volt_gles::glTexEnvi(target, pname, param)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glTexEnvfv(target: GLenum, pname: GLenum, params: *const GLfloat) {
    volt_gles::glTexEnvfv(target, pname, params)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glEnable(cap: GLenum) {
    volt_gles::glEnable(cap)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glDisable(cap: GLenum) {
    volt_gles::glDisable(cap)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glIsEnabled(cap: GLenum) -> GLboolean {
    volt_gles::glIsEnabled(cap)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glAlphaFunc(func: GLenum, ref_val: GLclampf) {
    volt_gles::glAlphaFunc(func, ref_val)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glBlendFunc(sfactor: GLenum, dfactor: GLenum) {
    volt_gles::glBlendFunc(sfactor, dfactor)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glBlendColor(
    red: GLclampf,
    green: GLclampf,
    blue: GLclampf,
    alpha: GLclampf,
) {
    volt_gles::glBlendColor(red, green, blue, alpha)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glBlendFuncSeparate(
    srcRGB: GLenum,
    dstRGB: GLenum,
    srcAlpha: GLenum,
    dstAlpha: GLenum,
) {
    volt_gles::glBlendFuncSeparate(srcRGB, dstRGB, srcAlpha, dstAlpha)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glDepthFunc(func: GLenum) {
    volt_gles::glDepthFunc(func)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glDepthMask(flag: GLboolean) {
    volt_gles::glDepthMask(flag)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glColorMask(
    red: GLboolean,
    green: GLboolean,
    blue: GLboolean,
    alpha: GLboolean,
) {
    volt_gles::glColorMask(red, green, blue, alpha)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glCullFace(mode: GLenum) {
    volt_gles::glCullFace(mode)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glFrontFace(mode: GLenum) {
    volt_gles::glFrontFace(mode)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glPolygonOffset(factor: GLfloat, units: GLfloat) {
    volt_gles::glPolygonOffset(factor, units)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glLineWidth(width: GLfloat) {
    volt_gles::glLineWidth(width)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glPointSize(size: GLfloat) {
    volt_gles::glPointSize(size)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glShadeModel(mode: GLenum) {
    volt_gles::glShadeModel(mode)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glColorMaterial(face: GLenum, mode: GLenum) {
    volt_gles::glColorMaterial(face, mode)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glFogf(pname: GLenum, param: GLfloat) {
    volt_gles::glFogf(pname, param)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glFogfv(pname: GLenum, params: *const GLfloat) {
    volt_gles::glFogfv(pname, params)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glFogi(pname: GLenum, param: GLint) {
    volt_gles::glFogi(pname, param)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glFogx(pname: GLenum, param: GLfixed) {
    volt_gles::glFogx(pname, param)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glFogxv(pname: GLenum, params: *const GLfixed) {
    volt_gles::glFogxv(pname, params)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glHint(target: GLenum, mode: GLenum) {
    volt_gles::glHint(target, mode)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glDepthRangef(n: GLclampf, f: GLclampf) {
    volt_gles::glDepthRangef(n, f)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glDepthRange(n: GLclampd, f: GLclampd) {
    volt_gles::glDepthRange(n, f)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glDepthRangex(n: GLclampx, f: GLclampx) {
    volt_gles::glDepthRangex(n, f)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glLightf(light: GLenum, pname: GLenum, param: GLfloat) {
    volt_gles::glLightf(light, pname, param)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glLightfv(light: GLenum, pname: GLenum, params: *const GLfloat) {
    volt_gles::glLightfv(light, pname, params)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glLightModelf(pname: GLenum, param: GLfloat) {
    volt_gles::glLightModelf(pname, param)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glLightModelfv(pname: GLenum, params: *const GLfloat) {
    volt_gles::glLightModelfv(pname, params)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glMaterialf(face: GLenum, pname: GLenum, param: GLfloat) {
    volt_gles::glMaterialf(face, pname, param)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glMaterialfv(face: GLenum, pname: GLenum, params: *const GLfloat) {
    volt_gles::glMaterialfv(face, pname, params)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glStencilFunc(func: GLenum, ref_val: GLint, mask: GLuint) {
    volt_gles::glStencilFunc(func, ref_val, mask)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glStencilMask(mask: GLuint) {
    volt_gles::glStencilMask(mask)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glStencilOp(fail: GLenum, zfail: GLenum, zpass: GLenum) {
    volt_gles::glStencilOp(fail, zfail, zpass)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glViewport(x: GLint, y: GLint, width: GLsizei, height: GLsizei) {
    volt_gles::glViewport(x, y, width, height)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glScissor(x: GLint, y: GLint, width: GLsizei, height: GLsizei) {
    volt_gles::glScissor(x, y, width, height)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glClearColor(
    red: GLclampf,
    green: GLclampf,
    blue: GLclampf,
    alpha: GLclampf,
) {
    volt_gles::glClearColor(red, green, blue, alpha)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glClearDepthf(depth: GLclampf) {
    volt_gles::glClearDepthf(depth)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glClearDepth(depth: GLclampd) {
    volt_gles::glClearDepth(depth)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glClearStencil(s: GLint) {
    volt_gles::glClearStencil(s)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glClear(mask: GLbitfield) {
    volt_gles::glClear(mask)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glPixelStorei(pname: GLenum, param: GLint) {
    volt_gles::glPixelStorei(pname, param)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glReadPixels(
    x: GLint,
    y: GLint,
    width: GLsizei,
    height: GLsizei,
    format: GLenum,
    type_: GLenum,
    pixels: *mut c_void,
) {
    volt_gles::glReadPixels(x, y, width, height, format, type_, pixels)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glFlush() {
    volt_gles::glFlush()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glFinish() {
    volt_gles::glFinish()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glGetIntegerv(pname: GLenum, params: *mut GLint) {
    volt_gles::glGetIntegerv(pname, params)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glGetFloatv(pname: GLenum, params: *mut GLfloat) {
    volt_gles::glGetFloatv(pname, params)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glGetBooleanv(pname: GLenum, params: *mut GLboolean) {
    volt_gles::glGetBooleanv(pname, params)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glGetString(name: GLenum) -> *const GLubyte {
    volt_gles::glGetString(name)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glGetError() -> GLenum {
    volt_gles::glGetError()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glPushAttrib(mask: GLbitfield) {
    volt_gles::glPushAttrib(mask)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glPopAttrib() {
    volt_gles::glPopAttrib()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glPushClientAttrib(mask: GLbitfield) {
    volt_gles::glPushClientAttrib(mask)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glPopClientAttrib() {
    volt_gles::glPopClientAttrib()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glCreateShader(shader_type: GLenum) -> GLuint {
    volt_gles::glCreateShader(shader_type)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glShaderSource(
    shader: GLuint,
    count: GLsizei,
    string: *const *const GLchar,
    length: *const GLint,
) {
    volt_gles::glShaderSource(shader, count, string, length)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glCompileShader(shader: GLuint) {
    volt_gles::glCompileShader(shader)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glGetShaderiv(shader: GLuint, pname: GLenum, params: *mut GLint) {
    volt_gles::glGetShaderiv(shader, pname, params)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glGetShaderInfoLog(
    shader: GLuint,
    buf_size: GLsizei,
    length: *mut GLsizei,
    info_log: *mut GLchar,
) {
    volt_gles::glGetShaderInfoLog(shader, buf_size, length, info_log)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glDeleteShader(shader: GLuint) {
    volt_gles::glDeleteShader(shader)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glCreateProgram() -> GLuint {
    volt_gles::glCreateProgram()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glAttachShader(program: GLuint, shader: GLuint) {
    volt_gles::glAttachShader(program, shader)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glDetachShader(program: GLuint, shader: GLuint) {
    volt_gles::glDetachShader(program, shader)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glLinkProgram(program: GLuint) {
    volt_gles::glLinkProgram(program)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glGetProgramiv(program: GLuint, pname: GLenum, params: *mut GLint) {
    volt_gles::glGetProgramiv(program, pname, params)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glGetProgramInfoLog(
    program: GLuint,
    buf_size: GLsizei,
    length: *mut GLsizei,
    info_log: *mut GLchar,
) {
    volt_gles::glGetProgramInfoLog(program, buf_size, length, info_log)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glUseProgram(program: GLuint) {
    volt_gles::glUseProgram(program)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glDeleteProgram(program: GLuint) {
    volt_gles::glDeleteProgram(program)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glGetUniformLocation(program: GLuint, name: *const GLchar) -> GLint {
    volt_gles::glGetUniformLocation(program, name)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glGetAttribLocation(program: GLuint, name: *const GLchar) -> GLint {
    volt_gles::glGetAttribLocation(program, name)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glUniform1f(location: GLint, v0: GLfloat) {
    volt_gles::glUniform1f(location, v0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glUniform2f(location: GLint, v0: GLfloat, v1: GLfloat) {
    volt_gles::glUniform2f(location, v0, v1)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glUniform3f(location: GLint, v0: GLfloat, v1: GLfloat, v2: GLfloat) {
    volt_gles::glUniform3f(location, v0, v1, v2)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glUniform4f(
    location: GLint,
    v0: GLfloat,
    v1: GLfloat,
    v2: GLfloat,
    v3: GLfloat,
) {
    volt_gles::glUniform4f(location, v0, v1, v2, v3)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glUniform1i(location: GLint, v0: GLint) {
    volt_gles::glUniform1i(location, v0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glUniform2i(location: GLint, v0: GLint, v1: GLint) {
    volt_gles::glUniform2i(location, v0, v1)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glUniform3i(location: GLint, v0: GLint, v1: GLint, v2: GLint) {
    volt_gles::glUniform3i(location, v0, v1, v2)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glUniform4i(location: GLint, v0: GLint, v1: GLint, v2: GLint, v3: GLint) {
    volt_gles::glUniform4i(location, v0, v1, v2, v3)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glUniformMatrix4fv(
    location: GLint,
    count: GLsizei,
    transpose: GLboolean,
    value: *const GLfloat,
) {
    volt_gles::glUniformMatrix4fv(location, count, transpose, value)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glGenBuffers(n: GLsizei, buffers: *mut GLuint) {
    volt_gles::glGenBuffers(n, buffers)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glGenBuffersARB(n: GLsizei, buffers: *mut GLuint) {
    volt_gles::glGenBuffersARB(n, buffers)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glBindBuffer(target: GLenum, buffer: GLuint) {
    volt_gles::glBindBuffer(target, buffer)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glBindBufferARB(target: GLenum, buffer: GLuint) {
    volt_gles::glBindBufferARB(target, buffer)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glBufferData(
    target: GLenum,
    size: GLsizeiptr,
    data: *const c_void,
    usage: GLenum,
) {
    volt_gles::glBufferData(target, size, data, usage)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glBufferDataARB(
    target: GLenum,
    size: GLsizeiptr,
    data: *const c_void,
    usage: GLenum,
) {
    volt_gles::glBufferDataARB(target, size, data, usage)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glBufferSubData(
    target: GLenum,
    offset: GLintptr,
    size: GLsizeiptr,
    data: *const c_void,
) {
    volt_gles::glBufferSubData(target, offset, size, data)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glDeleteBuffers(n: GLsizei, buffers: *const GLuint) {
    volt_gles::glDeleteBuffers(n, buffers)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glDeleteBuffersARB(n: GLsizei, buffers: *const GLuint) {
    volt_gles::glDeleteBuffersARB(n, buffers)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glVertexAttribPointer(
    index: GLuint,
    size: GLint,
    type_: GLenum,
    normalized: GLboolean,
    stride: GLsizei,
    pointer: *const c_void,
) {
    volt_gles::glVertexAttribPointer(index, size, type_, normalized, stride, pointer)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glEnableVertexAttribArray(index: GLuint) {
    volt_gles::glEnableVertexAttribArray(index)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glDisableVertexAttribArray(index: GLuint) {
    volt_gles::glDisableVertexAttribArray(index)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glGenFramebuffers(n: GLsizei, framebuffers: *mut GLuint) {
    volt_gles::glGenFramebuffers(n, framebuffers)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glBindFramebuffer(target: GLenum, framebuffer: GLuint) {
    volt_gles::glBindFramebuffer(target, framebuffer)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glFramebufferTexture2D(
    target: GLenum,
    attachment: GLenum,
    textarget: GLenum,
    texture: GLuint,
    level: GLint,
) {
    volt_gles::glFramebufferTexture2D(target, attachment, textarget, texture, level)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glDeleteFramebuffers(n: GLsizei, framebuffers: *const GLuint) {
    volt_gles::glDeleteFramebuffers(n, framebuffers)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glCheckFramebufferStatus(target: GLenum) -> GLenum {
    volt_gles::glCheckFramebufferStatus(target)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glGenRenderbuffers(n: GLsizei, renderbuffers: *mut GLuint) {
    volt_gles::glGenRenderbuffers(n, renderbuffers)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glBindRenderbuffer(target: GLenum, renderbuffer: GLuint) {
    volt_gles::glBindRenderbuffer(target, renderbuffer)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glRenderbufferStorage(
    target: GLenum,
    internalformat: GLenum,
    width: GLsizei,
    height: GLsizei,
) {
    volt_gles::glRenderbufferStorage(target, internalformat, width, height)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glDeleteRenderbuffers(n: GLsizei, renderbuffers: *const GLuint) {
    volt_gles::glDeleteRenderbuffers(n, renderbuffers)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glGenerateMipmap(target: GLenum) {
    volt_gles::glGenerateMipmap(target)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glGenQueries(n: GLsizei, ids: *mut GLuint) {
    volt_gles::glGenQueries(n, ids)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glGenQueriesARB(n: GLsizei, ids: *mut GLuint) {
    volt_gles::glGenQueriesARB(n, ids)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glBeginQuery(target: GLenum, id: GLuint) {
    volt_gles::glBeginQuery(target, id)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glBeginQueryARB(target: GLenum, id: GLuint) {
    volt_gles::glBeginQueryARB(target, id)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glEndQuery(target: GLenum) {
    volt_gles::glEndQuery(target)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glEndQueryARB(target: GLenum) {
    volt_gles::glEndQueryARB(target)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glGetQueryObjectuiv(id: GLuint, pname: GLenum, params: *mut GLuint) {
    volt_gles::glGetQueryObjectuiv(id, pname, params)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glGetQueryObjectuivARB(id: GLuint, pname: GLenum, params: *mut GLuint) {
    volt_gles::glGetQueryObjectuivARB(id, pname, params)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glDeleteQueries(n: GLsizei, ids: *const GLuint) {
    volt_gles::glDeleteQueries(n, ids)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glDeleteQueriesARB(n: GLsizei, ids: *const GLuint) {
    volt_gles::glDeleteQueriesARB(n, ids)
}
