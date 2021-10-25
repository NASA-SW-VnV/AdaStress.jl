
"""
Pilot response object. Limits pilot compliance with CAS instructions.
"""
Base.@kwdef mutable struct PilotResponse
    compliant::Bool = true       # whether pilot complies with CAS advisories
	turn_rate::R    = 3.0        # complying turn rate
	vert_rate::R    = 10.0       # complying vertical rate
	cmd::Command    = Command()  # resultant effective command
end

function initialize(pr::PilotResponse)
	pr.cmd = Command()
end

function update(pr::PilotResponse, cmd::Command, cas::CAS)
	pr.cmd = deepcopy(cmd)
    pr.compliant || return

	# horizontal
	if cas.hra == :cw
		pr.cmd.psi_d = pr.turn_rate
	elseif cas.hra == :ccw
		pr.cmd.psi_d = -pr.turn_rate
	end

	# vertical
	if cas.vra == :asc
		pr.cmd.h_d = pr.vert_rate
	elseif cas.vra == :desc
		pr.cmd.h_d = -pr.vert_rate
	end
end
