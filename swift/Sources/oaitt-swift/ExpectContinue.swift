// Answer `Expect: 100-continue` so clients do not sit through their own timeout.
//
// Hummingbird does not handle the header, and curl sends it for any sizeable upload:
// without a reply it waits a full second before sending the body. A 4.4 MB request took
// 1.03s of which 1.00s was that wait.

import HTTPTypes
import Hummingbird
import NIOCore
import NIOHTTPTypes

final class ExpectContinueHandler: ChannelInboundHandler, RemovableChannelHandler {
    typealias InboundIn = HTTPRequestPart
    typealias InboundOut = HTTPRequestPart
    typealias OutboundOut = HTTPResponsePart

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        if case .head(let head) = unwrapInboundIn(data),
            head.headerFields[.expect]?.lowercased() == "100-continue"
        {
            context.writeAndFlush(
                wrapOutboundOut(.head(HTTPResponse(status: .continue))), promise: nil)
        }
        context.fireChannelRead(data)
    }
}
