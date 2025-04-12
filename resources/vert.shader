#version 330 core

layout (location = 0) in vec3 in_vertex;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_uv;

out vec3 v_vertex;
out vec3 v_normal;
out vec3 v_tangent;
out vec3 v_bitangent;

out vec2 v_uv;
out vec3 v_worldPos;

uniform mat4 camera;


void calculateBasis(out vec3 tangent, out vec3 bitangent, in vec3 normal)
{
	bitangent = vec3(0.0, 1.0, 0.0);

	float normalDotUp = dot(normal, bitangent);

	if (normalDotUp == 1.0)
	{
		bitangent = vec3(0.0, 0.0, -1.0);
	}
	else if (normalDotUp == -1.0)
	{
		bitangent = vec3(0.0, 0.0, 1.0);
	}
	
	tangent = cross(bitangent, normal);	
	bitangent = cross(normal, tangent);
} 

void main() {
    
	
	vec4 vert = camera * vec4(in_vertex, 1.0);
	//vert =  vec4(in_vertex+vec3(0,0,-1.5), 1.0);

	v_vertex = vert.xyz;
    v_normal = in_normal;
    v_uv = in_uv;
    v_worldPos = v_vertex;
    calculateBasis(v_tangent, v_bitangent, v_normal);

    gl_Position =  vert;
}